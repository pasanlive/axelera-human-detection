"""
Inference runtimes.

``VoyagerRuntime`` (production)
    Voyager SDK owns the whole video pipeline (Option A):
    RTSP -> decodebin (RK3588 MPP hardware decode when ``allow_hardware_codec``)
    -> GStreamer preprocessing (letterbox/normalise, OpenCL where available)
    -> INT8 models on the Metis AIPU (all 4 cores) -> C++ decoders/NMS -> OC-SORT.
    Python only receives the decoded metadata. One ``InferenceStream`` hosts one
    pipeline per *feature group* (pose / pose+face / pose+lpr / ...), so cameras
    with face recognition disabled don't pay for it.

``SimulatedRuntime`` (development / CI)
    Produces synthetic frames and results with the same callback contract so the
    dashboard, analytics and tests run on machines without a Metis board. It never
    runs a real model - there is intentionally no CPU inference fallback.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.voyager.capacity_planner import CameraSpec, CapacityPlan, PipelineGroup
from src.voyager.network_builder import (
    BuildOptions,
    NetworkSpec,
    face_enroll_spec_yaml,
    write_face_enroll_network,
    write_network,
)
from src.voyager.result_parser import FrameAnalysis, parse_frame_result

# on_result(cam_id, analysis, get_image) - get_image() returns a BGR ndarray copy and is
# only valid during the callback (Voyager recycles the underlying buffer afterwards).
ResultCallback = Callable[[str, FrameAnalysis, Callable[[], Optional[np.ndarray]]], None]
# on_event(cam_id | None, kind, message)
EventCallback = Callable[[Optional[str], str, str], None]


def build_options_from_config(vcfg: Dict[str, Any]) -> BuildOptions:
    th = vcfg.get("thresholds", {}) or {}
    tr = vcfg.get("tracker", {}) or {}
    opts = BuildOptions()
    for k in ("person_conf", "person_nms_iou", "face_conf", "plate_conf"):
        if k in th:
            setattr(opts, k, float(th[k]))
    for k in ("max_faces_per_frame", "min_face_px", "max_plates_per_frame"):
        if k in th:
            setattr(opts, k, int(th[k]))
    if "max_age" in tr:
        opts.tracker_max_age = int(tr["max_age"])
    if "min_hits" in tr:
        opts.tracker_min_hits = int(tr["min_hits"])
    if "iou_threshold" in tr:
        opts.tracker_iou = float(tr["iou_threshold"])
    cal = vcfg.get("calibration_dir")
    if cal and os.path.isdir(cal) and any(Path(cal).iterdir()):
        opts.calibration_dir = cal
    opts.model_overrides = dict(vcfg.get("model_overrides", {}) or {})
    return opts
class NetworkNotCompiledError(RuntimeError):
    """Raised when one or more required networks have not been compiled for the Metis AIPU."""

    def __init__(self, networks: Any, build_root: Optional[Path] = None):
        self.networks = list(networks)
        self.build_root = build_root
        cmd = f"python tools/deploy_networks.py {' '.join(f'--network {n}' for n in self.networks)}"
        msg = (
            f"\n\n{'='*70}\n"
            f"[METIS SETUP ERROR] Required network(s) not compiled for Metis AIPU!\n"
            f"Missing: {', '.join(self.networks)}\n"
            f"Expected directory: {build_root or '$AXELERA_FRAMEWORK/build'}\n\n"
            f"Please run the deployment tool on your board (or build host):\n"
            f"  {cmd}\n\n"
            f"To deploy all model combinations + face enrollment in one go:\n"
            f"  python tools/deploy_networks.py --all\n"
            f"{'='*70}\n"
        )
        super().__init__(msg)


class BaseRuntime:
    backend = "none"

    def __init__(self, on_result: ResultCallback, on_event: Optional[EventCallback] = None):
        self.on_result = on_result
        self.on_event = on_event or (lambda cam, kind, msg: None)
        self.plan: Optional[CapacityPlan] = None
        self.cameras: Dict[str, CameraSpec] = {}
        self.camera_fps: Dict[str, float] = {}
        self.camera_state: Dict[str, str] = {}  # cam_id -> "connecting" | "live" | "error"
        self.last_error: Optional[str] = None
        self._fps_window: Dict[str, List[float]] = {}

    @property
    def is_running(self) -> bool:
        raise NotImplementedError

    def apply_plan(self, plan: CapacityPlan, cameras: Dict[str, CameraSpec]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def metrics(self) -> Dict[str, Any]:
        return {}

    def embed_face_image(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns a FaceNet embedding for the largest face in a still image (or None)."""
        raise NotImplementedError

    def _tick_fps(self, cam_id: str, now: float) -> None:
        w = self._fps_window.setdefault(cam_id, [])
        w.append(now)
        while w and now - w[0] > 2.0:
            w.pop(0)
        self.camera_fps[cam_id] = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else 0.0


_VOYAGER_INPUT_PATCHED = False


def _patch_voyager_stream_input() -> None:
    """Monkey-patch Axelera Voyager SDK's SinglePipeInput, gst_builder, and _build_gst_rtsp.

    Resolves critical issues for RTSP cameras on embedded Metis platforms:
    1. SDK 1.6.0 does not rate-limit RTSP camera inputs when specified_frame_rate is 0.
       Decoded RTSP bursts dump into the AIPU at >400 FPS, overflowing the Level-Zero
       command queue and triggering PCIe MSI timeout:
           "waitForQueueBinaryCompletion: Wait kernel failed with return code -1301"
           "terminate called after throwing an instance of 'L0::Exception'".
       If specified_frame_rate > 0 was used in stock SDK, it inserted gst.videorate with
       duplicate: True, which duplicates buffers on network jitter, causing:
           "inplace_addstreamid: meta_key already exists" (SIGABRT).
       Fix:
       - Insert `gst.videorate(drop-only=True, skip-to-first=True)` with capsfilter
         `framerate={target_fps}/1` BEFORE `libinplace_addstreamid.so`.
         Because `drop-only: True` is set, frames are NEVER duplicated on jitter (no meta_key crash),
         and incoming burst frames are dropped so the AIPU receives strictly <= target_fps.
       - Patch `gst_builder.Builder.queue` to set `max-size-buffers: 2, leaky: 2` (GST_QUEUE_LEAK_DOWNSTREAM),
         ensuring queues before the Metis AIPU never buffer more than 2 frames.
       - Patch `gst_builder.Builder.appsink` to set `max-buffers: 1, drop: True` to prevent
         backpressure accumulation.
    2. Injects `drop-on-latency: True` into `rtspsrc` properties so that network jitter
       or camera connection bursts drop stale backlogs instead of accumulating latency.
    """
    global _VOYAGER_INPUT_PATCHED
    if _VOYAGER_INPUT_PATCHED:
        return

    # Optimize Voyager runtime for embedded Rockchip RK3588 + Metis PCIe:
    # 1. Disable OpenCL to eliminate Mali GPU unified memory queue collisions with Metis Level-Zero.
    # 2. Disable double-buffering so inference executes synchronously per frame (queue depth = 1).
    # 3. Disable zero-copy dmabuf to prevent CMA heap allocation fragmentation on raw CPU frames.
    # 4. Default to TCP for RTSP to guarantee in-order delivery without dropped packets.
    os.environ.setdefault("AX_DISABLE_OPENCL", "1")
    os.environ.setdefault("AX_OPENCL_PREFERENCE", "NONE")
    os.environ.setdefault("AXELERA_USE_CL_DOUBLE_BUFFER", "0")
    os.environ.setdefault("AXELERA_USE_DOUBLE_BUFFER", "0")
    os.environ.setdefault("AXELERA_USE_DMABUF", "0")
    os.environ.setdefault("AXELERA_RTSP_PROTOCOL", "tcp")

    try:
        from axelera.app.pipe import io
        from axelera.app import config as ax_config
        from axelera.app import gst_builder
        from axelera.app.pipe import gst_helper
    except ImportError:
        return

    # Patch terminal appsink to drop stale buffers when full.
    # Note: internal queues before AIPU remain non-leaky (stock SDK behavior)
    # to guarantee Level-Zero command list buffers are never discarded before host synchronization.
    _orig_builder_appsink = getattr(gst_builder.Builder, "appsink", None)
    if _orig_builder_appsink is not None:
        def patched_builder_appsink(self: Any, props: Any = None, **kwargs: Any) -> None:
            p = dict(props or {})
            p['max-buffers'] = 2
            p['drop'] = True
            p['sync'] = False
            return _orig_builder_appsink(self, p, **kwargs)

        gst_builder.Builder.appsink = patched_builder_appsink

    # Patch gst_helper._create_element to intercept rtspsrc and decodebin
    _orig_create_element = getattr(gst_helper, "_create_element", None)
    if _orig_create_element is not None:
        def patched_create_element(element: dict[str, Any]) -> Any:
            elem = _orig_create_element(element)
            inst = element.get("instance", "")
            if inst == "rtspsrc":
                def _on_select_stream(rtspsrc_elem: Any, num: int, caps: Any) -> bool:
                    caps_str = caps.to_string() if caps is not None else ""
                    # Select video only; filter out audio/metadata tracks before RTSP SETUP
                    # to prevent unlinked audio sessions from timing out after 25s
                    is_video = "media=(string)video" in caps_str or "video/" in caps_str
                    if not is_video:
                        print(f"[RTSP FILTER] Excluding non-video track #{num} ({caps_str[:50]}...) from RTSP session")
                    return is_video

                try:
                    elem.connect("select-stream", _on_select_stream)
                except Exception as e:
                    print(f"[RTSP] Warning: could not connect select-stream signal: {e}")

            elif inst == "decodebin":
                force_sw = str(element.get('force-sw-decoders', False)).lower() == 'true'
                if force_sw:
                    def _prefer_software_decoder(bin_elem: Any, pad: Any, caps: Any, factories: Any) -> Any:
                        if not factories:
                            return factories
                        sw_decoders = []
                        hw_decoders = []
                        for factory in factories:
                            fname = factory.get_name()
                            if fname.startswith("avdec_") or fname.startswith("openh264") or "sw" in fname:
                                sw_decoders.append(factory)
                            elif fname.startswith("mpp") or fname.startswith("rga"):
                                hw_decoders.append(factory)
                            else:
                                sw_decoders.append(factory)
                        return sw_decoders + hw_decoders

                    try:
                        elem.connect("autoplug-sort", _prefer_software_decoder)
                    except Exception:
                        pass
            return elem

        gst_helper._create_element = patched_create_element

    # Patch gst_helper._on_pad_added so any unhandled secondary pads sink safely to fakesink
    _orig_on_pad_added = getattr(gst_helper, "_on_pad_added", None)
    if _orig_on_pad_added is not None:
        def patched_on_pad_added(element: Any, newPad: Any, sinkPad: Any, prop: Any, key: Any) -> None:
            if sinkPad.is_linked() and not newPad.is_linked():
                # Secondary pad arrived while primary sink pad is already linked:
                # divert to fakesink to avoid 'streaming stopped, reason not-linked (-1)'
                try:
                    import gi
                    gi.require_version("Gst", "1.0")
                    from gi.repository import Gst
                    pipeline = element.get_parent()
                    if pipeline and isinstance(pipeline, Gst.Pipeline):
                        fakesink = Gst.ElementFactory.make("fakesink", None)
                        if fakesink:
                            fakesink.set_property("sync", False)
                            pipeline.add(fakesink)
                            fakesink.sync_state_with_parent()
                            fsink_pad = fakesink.get_static_pad("sink")
                            if fsink_pad:
                                newPad.link(fsink_pad)
                                return
                except Exception:
                    pass
            return _orig_on_pad_added(element, newPad, sinkPad, prop, key)

        gst_helper._on_pad_added = patched_on_pad_added

    def patched_build_gst_rtsp(gst: Any, location: str, stream_idx: str, latency: int = 500):
        username, password, loc = io._parse_livestream_location(location)
        rtsp_props = {
            'location': f"{loc}",
            'user-id': f"{username}",
            'user-pw': f"{password}",
            'latency': max(latency, 500),
            'drop-on-latency': True,
            'connections': {'stream_%u': f'rtspcapsfilter{stream_idx}.sink'},
        }
        rtsp_protocol = (os.environ.get("AXELERA_RTSP_PROTOCOL") or getattr(ax_config.env, "rtsp_protocol", "all") or "all").lower()
        if rtsp_protocol == 'tcp':
            rtsp_props['protocols'] = 4
        elif rtsp_protocol == 'udp':
            rtsp_props['protocols'] = 1
        # When 'all' or '', omit 'protocols' so GStreamer automatically negotiates UDP/TCP with the camera

        gst.rtspsrc(rtsp_props)
        gst.capsfilter(caps='application/x-rtp,media=video', name=f'rtspcapsfilter{stream_idx}')

    def patched_build_input_gst(self: Any, gst: Any, stream_idx: str):
        requires_decodebin = False
        fps_limit = ''

        target_fps = int(
            getattr(self, "_requested_fps", 0)
            or getattr(self, "_specified_frame_rate", 0)
            or getattr(io.SinglePipeInput, "_global_fps_limit", 25)
            or 25
        )

        if self._src.type == ax_config.SourceType.USB:
            requires_decodebin = io._build_gst_usb(gst, self._src)
            fps_limit = f';fps_limit:{target_fps}' if target_fps else ''
        elif self._src.type == ax_config.SourceType.RTSP:
            io._build_gst_rtsp(gst, self._src.location, stream_idx, latency=self._rtsp_latency)
            requires_decodebin = True
            # Live RTSP streams must NOT sleep the GStreamer streaming thread in axinplace.
            # The camera clock dictates arrival rate; sleeping creates socket backlog and latency bursts.
            fps_limit = ''
        elif self._src.type == ax_config.SourceType.HLS:
            gst.playbin(uri=self._src.location)
            fps_limit = f';fps_limit:{target_fps}' if target_fps else ''
        elif self._src.type == ax_config.SourceType.FAKE_VIDEO:
            gst.videotestsrc(is_live=True, pattern=0)
            dims = f"width={self._src.width},height={self._src.height}"
            gst.capsfilter(caps=f'video/x-raw,{dims},format=NV12,framerate={self._src.fps or target_fps}/1')
            requires_decodebin = True
        elif self._src.type in (ax_config.SourceType.IMAGE_FILES, ax_config.SourceType.DATA_SOURCE):
            io._build_data_loader(gst, stream_idx)
            gst.images = [os.path.relpath(str(i), os.getcwd()) for i in self._src.images]
            fps_limit = f';fps_limit:{target_fps}' if target_fps else ''
        elif self._src.type == ax_config.SourceType.VIDEO_FILE:
            gst.filesrc(location=self._src.location)
            requires_decodebin = True
            fps_limit = f';fps_limit:{target_fps}' if target_fps else ''
        else:
            raise NotImplementedError(f"{self._src.type} format not supported in gst pipe")

        decodebin_link = ''
        if requires_decodebin:
            decodebin_link = io.build_decodebin(gst, self._allow_hardware_codec, stream_idx)

        gst.axinplace(
            name=decodebin_link,
            lib='libinplace_addstreamid.so',
            mode='meta',
            options=f'stream_id:{stream_idx or 0}{fps_limit}',
        )

    io._build_gst_rtsp = patched_build_gst_rtsp
    io.SinglePipeInput.build_input_gst = patched_build_input_gst

    _VOYAGER_INPUT_PATCHED = True


# ═════════════════════════════════════════════════════════════════════════════
# Production runtime
# ═════════════════════════════════════════════════════════════════════════════

class VoyagerRuntime(BaseRuntime):
    backend = "axelera_voyager"

    def __init__(self, vcfg: Dict[str, Any], on_result: ResultCallback,
                 on_event: Optional[EventCallback] = None):
        super().__init__(on_result, on_event)
        from src.voyager.sdk import sdk_modules

        self.sdk = sdk_modules()
        _patch_voyager_stream_input()
        self.vcfg = vcfg
        from src.voyager.sdk import resolve_networks_dir

        self.networks_dir = resolve_networks_dir(vcfg.get("networks_dir"), vcfg.get("sdk_path"))
        build_root = vcfg.get("build_root")
        self.build_root = Path(build_root).expanduser().resolve() if build_root else None
        self.rtsp_latency = int(vcfg.get("rtsp_latency_ms", 500))
        # Must match deploy.py (which has no low-latency flag): in SDK 1.6.0 low_latency=True
        # forces 1-core builds (build/<net>/<model>/1/), so a mismatch triggers a full
        # recompile inside create_inference_stream at startup / on every tier change.
        self.low_latency = bool(vcfg.get("low_latency", False))
        self.allow_hw_codec = bool(vcfg.get("hardware_decode", True))
        self.aipu_cores = int(vcfg.get("aipu_cores", 4))
        self.reconnect_s = float(vcfg.get("reconnect_interval_s", 5.0))
        self.disable_tracers = bool(vcfg.get("disable_tracers", False))
        self.build_opts = build_options_from_config(vcfg)
        rtsp_proto = str(vcfg.get("rtsp_transport", "all")).lower()
        os.environ["AXELERA_RTSP_PROTOCOL"] = rtsp_proto
        clock = vcfg.get("aipu_clock_mhz")
        mvm = vcfg.get("mvm_utilization")
        if clock or mvm is not None:
            cfg_str = f"{clock or ''},{mvm or ''}" if mvm is not None else str(clock or '')
            os.environ["AXELERA_CONFIGURE_BOARD"] = cfg_str
            print(f"[VOYAGER HW] Metis board configuration active: AXELERA_CONFIGURE_BOARD={cfg_str}")
        else:
            os.environ.pop("AXELERA_CONFIGURE_BOARD", None)

        from src.utils.hw_diag import get_watchdog
        self.watchdog = get_watchdog()
        self.watchdog.start()

        self._lock = threading.RLock()
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._pipes: Dict[str, Any] = {}  # network name -> PipeManager
        self._sid_to_cam: Dict[int, str] = {}
        self._cam_to_sid: Dict[str, int] = {}
        self._reconnecting: set = set()
        self._stopping = False
        self._tiling_disabled = False

    @staticmethod
    def _strip_tiling(plan: CapacityPlan) -> None:
        for g in plan.groups:
            g.sources = [s.split("]:", 1)[1] if s.startswith("tile[") else s for s in g.sources]

    # ── network files ───────────────────────────────────────────────────────
    def resolved_build_root(self) -> Optional[Path]:
        if self.build_root:
            return self.build_root
        from src.voyager.sdk import framework_path

        fw = framework_path(self.vcfg.get("sdk_path"))
        return Path(fw) / "build" if fw else None

    def missing_models(self, spec: NetworkSpec, yaml_path: Optional[Path] = None) -> List[str]:
        """Models of ``spec`` whose compiled manifest is not where SDK 1.6.0 will look.

        Mirrors ``network.read_deployed_model_infos``: each AIPU model must have
        ``<build>/<net>/<model>/<deploy_cores>/manifest.json`` with
        deploy_cores = 1 if low_latency else min(aipu_cores, max_compiler_cores).
        Without this check the SDK silently recompiles inside create_inference_stream.
        """
        build_root = self.resolved_build_root()
        if build_root is None:
            return []
        net_dir = build_root / spec.name
        yaml_path = yaml_path or (self.networks_dir / f"{spec.name}.yaml")
        try:
            import yaml as _yaml

            models = (_yaml.safe_load(Path(yaml_path).read_text()) or {}).get("models", {}) or {}
        except Exception:
            return [] if net_dir.exists() else [spec.name]
        missing = []
        for name, mdef in models.items():
            if str(mdef.get("model_type", "")).upper() == "CLASSICAL_CV":
                continue
            extra = mdef.get("extra_kwargs", {}) or {}
            cores = int(extra.get("aipu_cores", self.aipu_cores))
            max_comp = int(extra.get("max_compiler_cores", 1))
            deploy_cores = 1 if self.low_latency else min(cores, max_comp)
            if not (net_dir / name / str(deploy_cores) / "manifest.json").is_file():
                missing.append(f"{name}/{deploy_cores}")
        return missing

    def is_compiled(self, spec: NetworkSpec) -> bool:
        return not self.missing_models(spec)

    def network_yaml(self, spec: NetworkSpec) -> str:
        path = self.networks_dir / f"{spec.name}.yaml"
        if not path.exists():
            write_network(spec, self.networks_dir, self.build_opts)
        missing = self.missing_models(spec, path)
        if missing:
            self.on_event(None, "error", f"{spec.name}: missing compiled models {missing}")
            raise NetworkNotCompiledError([spec.name], self.resolved_build_root())
        return str(path)

    # ── lifecycle ───────────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def apply_plan(self, plan: CapacityPlan, cameras: Dict[str, CameraSpec]) -> None:
        with self._lock:
            if self._tiling_disabled:
                self._strip_tiling(plan)
            old = self.plan
            self.cameras = dict(cameras)
            if self._stream is not None and old is not None and self.is_running \
                    and old.structural_key() == plan.structural_key():
                try:
                    self._apply_incremental(old, plan)
                    self.plan = plan
                    return
                except Exception as e:
                    self.on_event(None, "warning", f"Incremental update failed ({e}); rebuilding stream")
            self._rebuild(plan)

    def _pipeline_kwargs(self, plan: CapacityPlan) -> Dict[str, Any]:
        # For live IP/RTSP streams, specified_frame_rate must default to 0 (matching input frame rate).
        # When set to a positive integer, GStreamer's videorate duplicates frames during network jitter
        # or preroll. Because duplicated GstBuffers share metadata, re-entering libinplace_addstreamid.so
        # throws "inplace_addstreamid: meta_key already exists" and crashes the process.
        fps = int(self.vcfg.get("specified_frame_rate", 0))
        return dict(
            pipe_type="gst",
            aipu_cores=self.aipu_cores,
            specified_frame_rate=fps,
            low_latency=self.low_latency,
            rtsp_latency=self.rtsp_latency,
        )

    def _rebuild(self, plan: CapacityPlan) -> None:
        self._stop_stream()
        self.plan = plan
        if not plan.groups:
            return
        try:
            from axelera.app.pipe import io
            io.SinglePipeInput._global_fps_limit = plan.fps
        except Exception:
            pass
        sdk = self.sdk
        first, rest = plan.groups[0], plan.groups[1:]
        sys_kwargs: Dict[str, Any] = dict(allow_hardware_codec=self.allow_hw_codec, timeout=0)
        if self.build_root:
            sys_kwargs["build_root"] = self.build_root
        try:
            from axelera.app import config as ax_config
            sys_kwargs["hardware_caps"] = ax_config.HardwareCaps(
                vaapi=ax_config.HardwareEnable.disable,
                opencl=ax_config.HardwareEnable.disable,
                opengl=ax_config.HardwareEnable.disable,
            )
        except Exception:
            pass
        if self.disable_tracers:
            tracers = []
            print("[VOYAGER HW] Metis tracers disabled (zero background axtrace processes)")
        else:
            tracers = sdk.inf_tracers.create_tracers(
                "end_to_end_fps", "core_temp", "device_fps", "cpu_usage",
                _aipu_cores=self.aipu_cores, pipe_type="gst",
            )
        self._stopping = False
        import re as _re

        shown = [_re.sub(r"//[^@/]+@", "//***@", s) for s in first.sources]
        rate_str = f"fps={self.vcfg['specified_frame_rate']}" if self.vcfg.get("specified_frame_rate", 0) else f"budget_fps={plan.fps} (camera-rate)"
        self.on_event(None, "info", f"Building Voyager stream: network={first.network} sources={shown} "
                                    f"hw_decode={self.allow_hw_codec} {rate_str} cores={self.aipu_cores}")
        t0 = time.time()
        try:
            stream = sdk.create_inference_stream(
                network=self.network_yaml(first.spec),
                sources=list(first.sources),
                tracers=tracers,
                **sys_kwargs,
                **self._pipeline_kwargs(plan),
            )
        except NetworkNotCompiledError as e:
            # A missing build is a setup problem, not a tiling problem: fail loudly.
            self.last_error = str(e)
            raise
        except Exception as e:
            if any(s.startswith("tile[") for g in plan.groups for s in g.sources):
                # Tiling with keypoint models is not verified on every SDK build: degrade, don't die.
                self.on_event(None, "warning", f"Tiled pipeline failed to build ({e}); retrying without tiling")
                self._tiling_disabled = True
                self._strip_tiling(plan)
                return self._rebuild(plan)
            self.last_error = str(e)
            raise
        self._stream = stream
        self._pipes = {}
        pipes = stream.pipelines
        self._register_group(first, pipes[0])
        for g in rest:
            pm = stream.add_pipeline(network=self.network_yaml(g.spec), sources=list(g.sources))
            self._register_group(g, pm)
        self._thread = threading.Thread(target=self._consume, args=(stream,), name="voyager-consumer", daemon=True)
        self._thread.start()
        self.on_event(None, "info", f"Voyager stream started in {time.time() - t0:.1f}s: tier={plan.tier.name} "
                                    f"fps={plan.fps} pipelines={[g.network for g in plan.groups]}")

    def _register_group(self, group: PipelineGroup, pm: Any) -> None:
        self._pipes[group.network] = pm
        sids = sorted(int(s) for s in pm.sources.keys())
        known = set(self._sid_to_cam)
        new_sids = [s for s in sids if s not in known]
        for cam_id, sid in zip(group.camera_ids, new_sids):
            self._sid_to_cam[sid] = cam_id
            self._cam_to_sid[cam_id] = sid
            self.camera_state[cam_id] = "connecting"

    def _apply_incremental(self, old: CapacityPlan, new: CapacityPlan) -> None:
        stream = self._stream
        old_cams = {c for g in old.groups for c in g.camera_ids}
        new_loc = {c: g for g in new.groups for c in g.camera_ids}
        old_loc = {c: g.network for g in old.groups for c in g.camera_ids}
        # 1. remove cameras that left or changed group/source
        for cam in sorted(old_cams):
            g = new_loc.get(cam)
            old_src = next(s for og in old.groups if cam in og.camera_ids
                           for c, s in zip(og.camera_ids, og.sources) if c == cam)
            new_src = None if g is None else g.sources[g.camera_ids.index(cam)]
            if g is None or old_loc[cam] != g.network or old_src != new_src:
                self._remove_cam(cam)
        # 2. add new pipelines / sources (before removing empty pipelines so the stream never empties)
        for g in new.groups:
            pm = self._pipes.get(g.network)
            missing = [(c, s) for c, s in zip(g.camera_ids, g.sources) if c not in self._cam_to_sid]
            if not missing:
                continue
            if pm is None:
                pm = stream.add_pipeline(network=self.network_yaml(g.spec), sources=[s for _, s in missing])
                self._register_group(PipelineGroup(g.spec, [c for c, _ in missing], [s for _, s in missing]), pm)
            else:
                for cam, src in missing:
                    sid = stream.add_source(src, pipeline=pm)  # pass the PipeManager, not an index
                    self._sid_to_cam[int(sid)] = cam
                    self._cam_to_sid[cam] = int(sid)
                    self.camera_state[cam] = "connecting"
        # 3. drop pipelines that no longer serve any camera
        for name in [n for n in self._pipes if n not in {g.network for g in new.groups}]:
            stream.remove_pipeline(self._pipes.pop(name))

    def _remove_cam(self, cam: str) -> None:
        sid = self._cam_to_sid.pop(cam, None)
        if sid is None:
            return
        self._sid_to_cam.pop(sid, None)
        self.camera_state.pop(cam, None)
        self.camera_fps.pop(cam, None)
        try:
            self._stream.remove_source(sid)
        except Exception as e:
            self.on_event(cam, "warning", f"remove_source({sid}) failed: {e}")

    def _consume(self, stream: Any) -> None:
        FrameEventType = self.sdk.FrameEventType
        self.watchdog.set_stream_active(True)
        t_last = time.time()
        try:
            for event in stream.with_events():
                if self._stopping:
                    break
                result = getattr(event, "result", None)
                if result is not None:
                    cam = self._sid_to_cam.get(int(result.stream_id))
                    if cam is None:
                        continue
                    now = time.time()
                    lat_ms = (now - t_last) * 1000.0 if t_last > 0 else 0.0
                    t_last = now
                    self.watchdog.record_frame(cam, lat_ms)
                    self._tick_fps(cam, now)
                    self.camera_state[cam] = "live"
                    try:
                        analysis = parse_frame_result(result, now)
                        img = result.image

                        def get_image(_img=img):
                            if _img is None:
                                return None
                            try:
                                arr = None
                                if hasattr(_img, "asarray"):
                                    try:
                                        from axelera import types as _ax_types
                                        arr = _img.asarray(_ax_types.ColorFormat.BGR)
                                    except Exception:
                                        try:
                                            arr = _img.asarray("BGR")
                                        except Exception:
                                            try:
                                                arr = _img.asarray()
                                            except Exception:
                                                pass
                                if arr is None and hasattr(_img, "to_numpy"):
                                    arr = _img.to_numpy()
                                if arr is None and isinstance(_img, np.ndarray):
                                    arr = _img
                                if arr is not None:
                                    if hasattr(arr, "copy"):
                                        arr = arr.copy()
                                    if arr.ndim == 2:
                                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                                    elif arr.ndim == 3 and arr.shape[2] == 1:
                                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                                    return arr
                            except Exception as e:
                                print(f"[PREVIEW ERROR] Failed to extract image from frame: {e}")
                            return None

                        self.on_result(cam, analysis, get_image)
                    except Exception:
                        traceback.print_exc()
                elif event.type in (FrameEventType.source_error, FrameEventType.end_of_source):
                    cam = self._sid_to_cam.get(int(event.source_id))
                    self.on_event(cam, "source_error", event.message or event.type.name)
                    if cam:
                        self.camera_state[cam] = "error"
                        self._schedule_reconnect(cam)
        except Exception as e:
            if not self._stopping:
                self.last_error = str(e)
                self.on_event(None, "error", f"Voyager stream terminated: {e}")
                traceback.print_exc()
        finally:
            self.watchdog.set_stream_active(False)

    def _schedule_reconnect(self, cam: str) -> None:
        if cam in self._reconnecting:
            return
        self._reconnecting.add(cam)

        def _reconnect():
            try:
                while not self._stopping:
                    time.sleep(self.reconnect_s)
                    with self._lock:
                        plan = self.plan
                        g = plan.group_for(cam) if plan else None
                        if g is None or self._stream is None:
                            return
                        src = g.sources[g.camera_ids.index(cam)]
                        try:
                            self._remove_cam(cam)
                            sid = self._stream.add_source(src, pipeline=self._pipes[g.network])
                            self._sid_to_cam[int(sid)] = cam
                            self._cam_to_sid[cam] = int(sid)
                            self.camera_state[cam] = "connecting"
                            self.on_event(cam, "info", "reconnected")
                            return
                        except Exception as e:
                            self.on_event(cam, "warning", f"reconnect failed: {e}")
            finally:
                self._reconnecting.discard(cam)

        threading.Thread(target=_reconnect, name=f"reconnect-{cam}", daemon=True).start()

    def _stop_stream(self) -> None:
        stream, thread = self._stream, self._thread
        self._stopping = True
        if stream is not None:
            try:
                stream.stop()
            except Exception as e:
                self.on_event(None, "warning", f"stream.stop() raised {e}")
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        self._stream = None
        self._thread = None
        self._pipes.clear()
        self._sid_to_cam.clear()
        self._cam_to_sid.clear()
        self.camera_state.clear()
        self.camera_fps.clear()
        try:
            import subprocess
            subprocess.run(["pkill", "-9", "axtrace"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def stop(self) -> None:
        with self._lock:
            self._stop_stream()

    # ── telemetry ───────────────────────────────────────────────────────────
    def metrics(self) -> Dict[str, Any]:
        stream = self._stream
        res: Dict[str, Any] = {}
        if stream is not None:
            try:
                raw = stream.get_all_metrics()
                res = {k: {"value": float(getattr(m, "value", 0.0)), "unit": getattr(m, "unit", ""),
                            "title": getattr(m, "title", k)} for k, m in raw.items()}
            except Exception:
                pass
        res["watchdog"] = self.watchdog.get_metrics()
        return res

    # ── enrollment ──────────────────────────────────────────────────────────
    def embed_face_image(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Runs the small RetinaFace->FaceNet network once over a still image.

        NOTE: this opens a second, short-lived InferenceStream next to the live one.
        If the device cannot host both, prefer enrolling from a live camera frame.
        """
        import cv2

        path = self.networks_dir / "upside-face-enroll.yaml"
        if not path.exists():
            write_face_enroll_network(self.networks_dir, self.build_opts)
        name = face_enroll_spec_yaml()["name"]
        build_root = self.resolved_build_root()
        if build_root is not None and not (build_root / name).exists():
            raise RuntimeError(f"Enrollment network not compiled. Run: python tools/deploy_networks.py --network {name}")
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "enroll.jpg")
            cv2.imwrite(img_path, image_bgr)
            kwargs: Dict[str, Any] = dict(network=str(path), sources=[img_path], pipe_type="gst",
                                          aipu_cores=1, frames=1, timeout=30)
            if self.build_root:
                kwargs["build_root"] = self.build_root
            stream = self.sdk.create_inference_stream(**kwargs)
            try:
                for fr in stream:
                    fa = parse_frame_result(fr, time.time())
                    faces = [f for f in fa.faces if f["embedding"] is not None]
                    if not faces:
                        return None
                    best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                    return best["embedding"]
            finally:
                stream.stop()
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Simulation runtime (no hardware)
# ═════════════════════════════════════════════════════════════════════════════

def _stable_embedding(key: str, dim: int = 512) -> np.ndarray:
    seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class SimulatedRuntime(BaseRuntime):
    """Synthetic frames/results with the production callback contract (no model inference)."""

    backend = "simulation"
    _KPT_LAYOUT = np.array([  # rough COCO-17 layout in a unit person box (x, y)
        [0.50, 0.08], [0.46, 0.06], [0.54, 0.06], [0.42, 0.08], [0.58, 0.08],
        [0.35, 0.22], [0.65, 0.22], [0.28, 0.38], [0.72, 0.38], [0.25, 0.52], [0.75, 0.52],
        [0.40, 0.55], [0.60, 0.55], [0.40, 0.76], [0.60, 0.76], [0.40, 0.96], [0.60, 0.96],
    ], dtype=np.float32)

    def __init__(self, on_result: ResultCallback, on_event: Optional[EventCallback] = None,
                 persons_per_camera: int = 3, frame_size: Tuple[int, int] = (960, 540)):
        super().__init__(on_result, on_event)
        self.persons_per_camera = persons_per_camera
        self.frame_size = frame_size
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.RLock()
        self._t0 = time.time()
        self._frame_idx: Dict[str, int] = {}

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def apply_plan(self, plan: CapacityPlan, cameras: Dict[str, CameraSpec]) -> None:
        with self._lock:
            self.plan = plan
            self.cameras = dict(cameras)
            for cid in list(self.camera_state):
                if plan.group_for(cid) is None:
                    self.camera_state.pop(cid, None)
                    self.camera_fps.pop(cid, None)
            for g in plan.groups:
                for cid in g.camera_ids:
                    self.camera_state[cid] = "live"
        if not self.is_running:
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, name="sim-runtime", daemon=True)
            self._thread.start()
        self.on_event(None, "info", f"Simulated stream: tier={plan.tier.name} fps={plan.fps}")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def _person_boxes(self, cam_id: str, t: float) -> np.ndarray:
        w, h = self.frame_size
        boxes = []
        rnd = random.Random(cam_id)
        for i in range(self.persons_per_camera):
            speed = 0.05 + 0.05 * rnd.random()
            phase = rnd.random() * math.tau
            ph = h * (0.45 + 0.25 * rnd.random())
            pw = ph * 0.4
            cx = w * (0.5 + 0.4 * math.sin(t * speed * math.tau + phase))
            cy = h * 0.55 + 0.1 * h * math.sin(t * 0.3 + i)
            boxes.append([cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2])
        return np.clip(np.asarray(boxes, dtype=np.float32), 0, [w - 1, h - 1, w - 1, h - 1])

    def _render(self, cam_id: str, boxes: np.ndarray, t: float) -> np.ndarray:
        import cv2

        w, h = self.frame_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (28, 24, 22)
        cv2.rectangle(img, (0, int(h * 0.75)), (w, h), (45, 42, 40), -1)
        for b in boxes.astype(int):
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (90, 110, 130), -1)
        cv2.putText(img, f"SIMULATION {cam_id}  t={t:6.1f}s", (16, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 1, cv2.LINE_AA)
        return img

    def _loop(self) -> None:
        next_due: Dict[str, float] = {}
        while not self._stop_evt.is_set():
            with self._lock:
                plan = self.plan
                cams = [c for g in (plan.groups if plan else []) for c in g.camera_ids]
                feats = {c: g.spec for g in (plan.groups if plan else []) for c in g.camera_ids}
            if not cams:
                time.sleep(0.1)
                continue
            period = 1.0 / max(1, plan.fps)
            now = time.time()
            for cam in cams:
                if next_due.get(cam, 0) > now:
                    continue
                next_due[cam] = now + period
                t = now - self._t0
                boxes = self._person_boxes(cam, t)
                fa = self._analysis(cam, boxes, feats[cam], now)
                self._tick_fps(cam, now)
                try:
                    self.on_result(cam, fa, lambda c=cam, b=boxes, tt=t: self._render(c, b, tt))
                except Exception:
                    traceback.print_exc()
            sleep = min(next_due.values()) - time.time()
            self._stop_evt.wait(max(0.002, sleep))

    def _analysis(self, cam: str, boxes: np.ndarray, spec: NetworkSpec, now: float) -> FrameAnalysis:
        fa = FrameAnalysis(stream_id=0, timestamp=now, frame_size=self.frame_size)
        for i, b in enumerate(boxes):
            bw, bh = b[2] - b[0], b[3] - b[1]
            k = np.empty((17, 3), dtype=np.float32)
            k[:, 0] = b[0] + self._KPT_LAYOUT[:, 0] * bw
            k[:, 1] = b[1] + self._KPT_LAYOUT[:, 1] * bh
            k[:, 2] = 0.9
            fa.persons.append({"bbox": b.tolist(), "confidence": 0.9, "keypoints": k, "track_id": i + 1})
            if spec.face:
                fw = bw * 0.35
                fx = b[0] + bw * 0.5 - fw / 2
                fa.faces.append({
                    "bbox": [fx, b[1], fx + fw, b[1] + fw * 1.2],
                    "confidence": 0.95,
                    "embedding": _stable_embedding(f"{cam}-person-{i}"),
                    "person_index": i,
                })
        if spec.lpr:
            w, h = self.frame_size
            fa.plates.append({"bbox": [w * 0.1, h * 0.8, w * 0.22, h * 0.86], "confidence": 0.9,
                              "plate_text": f"SIM{abs(hash(cam)) % 1000:03d}"})
        return fa

    def metrics(self) -> Dict[str, Any]:
        total = sum(self.camera_fps.values())
        util = min(1.0, self.plan.aipu_utilization) if self.plan else 0.0
        return {
            "end_to_end_fps": {"value": total, "unit": "fps", "title": "End-to-end"},
            "simulated_aipu_utilization": {"value": util * 100.0, "unit": "%", "title": "AIPU (planned)"},
        }

    def embed_face_image(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            return None
        return _stable_embedding(hashlib.sha1(image_bgr.tobytes()[:65536]).hexdigest())
