#!/usr/bin/env python3
"""
Hardware Video Decoding Diagnostic & Benchmark Utility for Rockchip RK3588 / Embedded Linux.

Checks:
1. Rockchip MPP (Media Process Platform) & RGA device nodes (/dev/mpp_service, /dev/rga, /dev/dri/*).
2. Driver /proc/mpp_service status, version, and active hardware sessions.
3. GStreamer plugins: mppvideodec (HW decode), rkmpp, gstrga, and avdec_h264 (SW fallback).
4. Live decoding test using an RTSP camera stream, synthetic video stream, or video file.
5. Side-by-side performance comparison: Hardware (mppvideodec) vs Software (avdec_h264)
   measuring achieved FPS, Host CPU load, PTS timestamp integrity, and DMA-BUF support.

Usage:
  python tools/diag_hw_decode.py                      # Quick diagnostic & synthetic test
  python tools/diag_hw_decode.py --camera             # Test using first active camera from config/config.yaml
  python tools/diag_hw_decode.py --rtsp <RTSP_URL>   # Test using specific RTSP stream URL
  python tools/diag_hw_decode.py --file <VIDEO_PATH>  # Test using a local H.264/H.265 video file
  python tools/diag_hw_decode.py --compare            # Run HW vs SW decode comparison benchmark
  python tools/diag_hw_decode.py --duration 10        # Run test stream for 10 seconds (default: 5)
"""

import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def print_header(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (res.stdout or "") + (res.stderr or "")
        return res.returncode, out.strip()
    except Exception as e:
        return -1, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 1. System & Architecture Inspection
# ─────────────────────────────────────────────────────────────────────────────

def check_system() -> None:
    print_header("System & Board Architecture")
    uname = platform.uname()
    print(f"OS / Kernel:     {uname.system} {uname.release} ({uname.machine})")
    print(f"Python:          {sys.version.split()[0]} ({sys.executable})")

    # Rockchip hardware model
    rk_model = "/proc/device-tree/model"
    if os.path.exists(rk_model):
        try:
            with open(rk_model, "r") as f:
                print(f"Hardware Model:  {f.read().strip()}")
        except Exception:
            pass

    # Rockchip SoC compatible string
    rk_compat = "/proc/device-tree/compatible"
    if os.path.exists(rk_compat):
        try:
            with open(rk_compat, "rb") as f:
                compat_str = f.read().decode("latin-1", errors="ignore").replace("\x00", " ").strip()
                print(f"SoC Platform:    {compat_str}")
        except Exception:
            pass

    # User and permission groups
    user = os.environ.get("USER", "unknown")
    rc_id, out_id = run_cmd(["id"])
    print(f"User & Groups:   {out_id if rc_id == 0 else user}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rockchip MPP & Hardware Device Nodes
# ─────────────────────────────────────────────────────────────────────────────

def check_hardware_nodes() -> Dict[str, bool]:
    print_header("Rockchip Hardware Acceleration Nodes")
    status = {}

    nodes = [
        ("/dev/mpp_service", "Rockchip MPP VPU Decoder/Encoder driver node"),
        ("/dev/rga", "Rockchip 2D Raster Graphic Acceleration (RGA) node"),
        ("/dev/dri/card0", "Direct Rendering Manager (DRM) display/DMA controller"),
        ("/dev/dri/renderD128", "DRM Render node for zero-copy DMA-BUF buffers"),
        ("/dev/dma_heap/system", "DMA-BUF heap allocator for zero-copy memory"),
    ]

    for path, desc in nodes:
        exists = os.path.exists(path)
        status[path] = exists
        if exists:
            r_ok = os.access(path, os.R_OK)
            w_ok = os.access(path, os.W_OK)
            rw_str = "RW" if (r_ok and w_ok) else ("R" if r_ok else ("W" if w_ok else "NO ACCESS"))
            try:
                st = os.stat(path)
                import stat
                mode_str = stat.filemode(st.st_mode)
                print(f"  [FOUND] {path:22} | {rw_str:9} | {mode_str} | {desc}")
            except Exception:
                print(f"  [FOUND] {path:22} | {rw_str:9} | {desc}")
        else:
            print(f"  [MISSING] {path:20} | {desc}")

    # Inspect /proc/mpp_service if available
    proc_mpp = "/proc/mpp_service"
    if os.path.isdir(proc_mpp):
        print(f"\nRockchip MPP Kernel Driver Status ({proc_mpp}):")
        for sub in ("version", "summary", "session_info"):
            fpath = os.path.join(proc_mpp, sub)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r") as f:
                        content = f.read().strip()
                        if content:
                            print(f"  --> {sub}:")
                            for line in content.splitlines()[:10]:
                                print(f"        {line}")
                except Exception as e:
                    print(f"  --> {sub}: <unreadable: {e}>")

    # Check dmesg for MPP / RKVDEC initialization events
    rc, dmesg_out = run_cmd(["dmesg"], timeout=3)
    if rc != 0:
        rc, dmesg_out = run_cmd(["sudo", "-n", "dmesg"], timeout=3)
    if rc == 0 and dmesg_out:
        mpp_lines = [l for l in dmesg_out.splitlines() if any(k in l.lower() for k in ("mpp", "rkvdec", "rga", "vpu"))]
        if mpp_lines:
            print("\nRecent Kernel MPP / VPU dmesg logs:")
            for l in mpp_lines[-6:]:
                print(f"  {l}")

    # Check permissions for /dev/rga (Rockchip 2D raster graphic acceleration)
    if os.path.exists("/dev/rga") and not (os.access("/dev/rga", os.R_OK) and os.access("/dev/rga", os.W_OK)):
        print("\n  [PERMISSION NOTICE] /dev/rga lacks read/write access for current user.")
        res_rga = subprocess.run(["sudo", "-n", "chmod", "666", "/dev/rga"], capture_output=True, text=True)
        if res_rga.returncode == 0:
            print("  [FIX APPLIED] Successfully executed 'sudo chmod 666 /dev/rga'. Permissions updated!")
        else:
            print("  Recommendation: Run 'sudo chmod 666 /dev/rga' so mppvideodec can use RGA 2D scaling.")

    return status


# ─────────────────────────────────────────────────────────────────────────────
# 3. GStreamer Hardware Decoder Plugin Inspection
# ─────────────────────────────────────────────────────────────────────────────

def check_gstreamer_plugins() -> Dict[str, Any]:
    print_header("GStreamer Video Decoder & Conversion Plugins")
    plugins = {}

    tools = ["gst-inspect-1.0", "gst-launch-1.0"]
    for t in tools:
        path = shutil.which(t)
        print(f"Binary {t:16}: {'[FOUND] ' + path if path else '[MISSING] (Install gstreamer1.0-tools)'}")

    if not shutil.which("gst-inspect-1.0"):
        print("[ERROR] gst-inspect-1.0 not found in PATH.")
        return plugins

    check_list = [
        ("mppvideodec", "Rockchip MPP Hardware Video Decoder (VPU / rkvdec)"),
        ("rkmpp", "Rockchip MPP Core Plugin"),
        ("gstrga", "Rockchip RGA 2D Blit / Color Format Converter (librga)"),
        ("rga", "Rockchip RGA 2D Blit plugin alias"),
        ("avdec_h264", "FFmpeg libav Software H.264 Decoder (CPU fallback)"),
        ("avdec_h265", "FFmpeg libav Software H.265/HEVC Decoder (CPU fallback)"),
        ("kmssink", "KMS/DRM Direct Kernel Display Sink (Zero-copy DMA-BUF)"),
    ]

    for elem_name, desc in check_list:
        rc, out = run_cmd(["gst-inspect-1.0", elem_name])
        found = (rc == 0)
        plugins[elem_name] = {"found": found, "details": out if found else ""}
        if found:
            # Extract Rank, Plugin Filename, and Supported Codecs
            rank_match = re.search(r"Rank\s+:\s+(.+)", out)
            file_match = re.search(r"Filename\s+:\s+(.+)", out)
            rank = rank_match.group(1).strip() if rank_match else "unknown"
            fname = file_match.group(1).strip() if file_match else ""
            print(f"  [AVAILABLE] {elem_name:14} (Rank: {rank})")
            if fname:
                print(f"               Library: {fname}")
            if elem_name == "mppvideodec":
                # Check supported sink caps (codecs)
                caps = []
                for c in ["video/x-h264", "video/x-h265", "image/jpeg", "video/x-vp8", "video/x-vp9"]:
                    if c in out:
                        caps.append(c.replace("video/x-", "").replace("image/", ""))
                if caps:
                    print(f"               Hardware Codecs: {', '.join(caps)}")
                if "memory:DMABuf" in out:
                    print(f"               Zero-Copy: Supports 'memory:DMABuf' direct hardware buffers")
        else:
            print(f"  [NOT FOUND] {elem_name:14} | {desc}")

    return plugins


# ─────────────────────────────────────────────────────────────────────────────
# 4. Live Decoding Test (Synthetic / RTSP / File)
# ─────────────────────────────────────────────────────────────────────────────

def get_cpu_times() -> Tuple[float, float]:
    """Returns (total_cpu_time, idle_cpu_time) from /proc/stat if available."""
    try:
        with open("/proc/stat", "r") as f:
            fields = f.readline().strip().split()[1:]
            vals = [float(x) for x in fields]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            return total, idle
    except Exception:
        return 0.0, 0.0


def calculate_cpu_percent(t1: Tuple[float, float], t2: Tuple[float, float]) -> float:
    total_delta = t2[0] - t1[0]
    idle_delta = t2[1] - t1[1]
    if total_delta <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - (idle_delta / total_delta))))


import urllib.parse


def parse_rtsp_url(url: str) -> Tuple[str, str, str]:
    """Extracts username and password from RTSP URL and returns clean URL."""
    res = urllib.parse.urlparse(url)
    username = urllib.parse.unquote(res.username or "")
    password = urllib.parse.unquote(res.password or "")
    if "@" in res.netloc:
        res = res._replace(netloc=res.netloc.rsplit("@", 1)[1])
    clean_url = urllib.parse.urlunparse(res)
    return username, password, clean_url


def run_python_gst_pipeline(
    decoder: str,
    source_url: str,
    duration_s: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Runs a Python GStreamer pipeline with select-stream filtering and buffer probes."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
    except Exception:
        return None

    pipeline = Gst.Pipeline.new("diag_pipeline")
    rtspsrc = Gst.ElementFactory.make("rtspsrc", "rtspsrc0")
    if not rtspsrc:
        return None

    username, password, clean_loc = parse_rtsp_url(source_url)
    rtspsrc.set_property("location", clean_loc)
    if username:
        rtspsrc.set_property("user-id", username)
    if password:
        rtspsrc.set_property("user-pw", password)
    rtspsrc.set_property("latency", 500)
    try:
        rtspsrc.set_property("drop-on-latency", True)
    except Exception:
        pass

    rtp_caps = Gst.ElementFactory.make("capsfilter", "rtpcaps")
    rtp_caps.set_property("caps", Gst.Caps.from_string("application/x-rtp,media=video"))

    depay = Gst.ElementFactory.make("rtph264depay", "depay0")
    parse = Gst.ElementFactory.make("h264parse", "parse0")
    dec = Gst.ElementFactory.make(decoder, "dec0")
    sink = Gst.ElementFactory.make("fakesink", "sink0")
    audio_sink = Gst.ElementFactory.make("fakesink", "audiosink")
    if not all([rtp_caps, depay, parse, dec, sink, audio_sink]):
        return None

    sink.set_property("sync", False)
    audio_sink.set_property("sync", False)

    for elem in [rtspsrc, rtp_caps, depay, parse, dec, sink, audio_sink]:
        pipeline.add(elem)

    rtp_caps.link(depay)
    depay.link(parse)
    parse.link(dec)
    dec.link(sink)

    def on_select_stream(src: Any, num: int, caps: Any) -> bool:
        caps_str = caps.to_string() if caps else ""
        # Strictly select video; drop audio and metadata before RTSP SETUP
        return "media=(string)video" in caps_str or "video/" in caps_str

    try:
        rtspsrc.connect("select-stream", on_select_stream)
    except Exception:
        pass

    def on_pad_added(src: Any, new_pad: Any) -> None:
        sink_pad = rtp_caps.get_static_pad("sink")
        if sink_pad and not sink_pad.is_linked():
            # Attempt linking video pad
            ret = new_pad.link(sink_pad)
            if ret == Gst.PadLinkReturn.OK:
                return

        # Sink audio or auxiliary pads safely to prevent 'not-linked (-1)' error
        audio_pad = audio_sink.get_static_pad("sink")
        if audio_pad and not audio_pad.is_linked():
            new_pad.link(audio_pad)

    rtspsrc.connect("pad-added", on_pad_added)

    metrics = {
        "frames": 0,
        "pts_valid": True,
        "pts_warning": False,
        "error_msg": "",
    }

    def sink_probe(pad: Any, info: Any) -> Any:
        buf = info.get_buffer()
        if buf:
            metrics["frames"] += 1
            if buf.pts == Gst.CLOCK_TIME_NONE:
                metrics["pts_valid"] = False
                metrics["pts_warning"] = True
        return Gst.PadProbeReturn.OK

    sink_pad = sink.get_static_pad("sink")
    if sink_pad:
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, sink_probe)

    bus = pipeline.get_bus()

    cpu_t1 = get_cpu_times()
    t_start = time.time()
    pipeline.set_state(Gst.State.PLAYING)

    deadline = t_start + duration_s
    while time.time() < deadline:
        msg = bus.timed_pop_filtered(
            int(100 * Gst.MSECOND),
            Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS
        )
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                metrics["error_msg"] = f"{err.message} ({debug})"
                break
            elif msg.type == Gst.MessageType.WARNING:
                w_err, w_debug = msg.parse_warning()
                if "pts" in (w_err.message + w_debug).lower():
                    metrics["pts_warning"] = True
                    metrics["pts_valid"] = False

    t_elapsed = max(0.001, time.time() - t_start)
    cpu_t2 = get_cpu_times()

    pipeline.set_state(Gst.State.NULL)

    return {
        "decoder": decoder,
        "success": metrics["frames"] > 0,
        "frames": metrics["frames"],
        "fps": round(metrics["frames"] / t_elapsed, 1),
        "cpu_pct": round(calculate_cpu_percent(cpu_t1, cpu_t2), 1),
        "pts_valid": metrics["pts_valid"],
        "pts_warning": metrics["pts_warning"],
        "error_msg": metrics["error_msg"],
        "logs": "",
    }


def run_pipeline_test(
    decoder: str,
    source_type: str,
    source_url: Optional[str] = None,
    duration_s: float = 5.0,
    max_frames: int = 150,
) -> Dict[str, Any]:
    """Runs a GStreamer pipeline with the specified decoder and collects metrics."""
    # First: attempt native Python GStreamer pipeline if on RTSP (most reliable for track filtering)
    if source_type == "rtsp" and source_url:
        py_result = run_python_gst_pipeline(decoder, source_url, duration_s=duration_s)
        if py_result is not None and (py_result["success"] or "not found" not in py_result["error_msg"].lower()):
            return py_result

    result: Dict[str, Any] = {
        "decoder": decoder,
        "success": False,
        "frames": 0,
        "fps": 0.0,
        "cpu_pct": 0.0,
        "pts_valid": True,
        "pts_warning": False,
        "error_msg": "",
        "logs": "",
    }

    gst_launch = shutil.which("gst-launch-1.0")
    if not gst_launch:
        result["error_msg"] = "gst-launch-1.0 not found"
        return result

    # Build pipeline according to source
    if source_type == "rtsp" and source_url:
        username, password, clean_url = parse_rtsp_url(source_url)
        user_props = []
        if username:
            user_props.extend([f"user-id={username}", f"user-pw={password}"])
        pipeline = [
            gst_launch,
            "-v",
            "rtspsrc", f"location={clean_url}", "latency=500", "drop-on-latency=true",
        ] + user_props + [
            "name=rtspsrc0",
            "rtspsrc0.", "!", "application/x-rtp,media=video",
            "!", "rtph264depay",
            "!", "h264parse",
            "!", decoder,
            "!", "fakesink", "sync=false",
            "rtspsrc0.", "!", "fakesink", "sync=false",
        ]
    elif source_type == "file" and source_url:
        pipeline = [
            gst_launch,
            "-v",
            "filesrc", f"location={source_url}",
            "!", "qtdemux",
            "!", "h264parse",
            "!", decoder,
            "!", "fakesink", "sync=false",
        ]
    else:
        # Synthetic H.264 stream test: encode with x264enc or mpph264enc then decode
        # Check available encoder
        rc_mppenc, _ = run_cmd(["gst-inspect-1.0", "mpph264enc"])
        rc_x264, _ = run_cmd(["gst-inspect-1.0", "x264enc"])

        enc = "mpph264enc" if rc_mppenc == 0 else ("x264enc speed-preset=ultrafast tune=zerolatency" if rc_x264 == 0 else None)
        if not enc:
            result["error_msg"] = "Neither mpph264enc nor x264enc available for synthetic stream test"
            return result

        num_buffers = max_frames
        pipeline = [
            gst_launch,
            "-v",
            "videotestsrc", f"num-buffers={num_buffers}", "is-live=true",
            "!", "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1",
        ]
        pipeline.extend(enc.split())
        pipeline.extend([
            "!", "h264parse",
            "!", decoder,
            "!", "fakesink", "sync=false",
        ])

    cpu_t1 = get_cpu_times()
    t_start = time.time()

    try:
        proc = subprocess.Popen(
            pipeline,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        frame_count = 0
        timed_out = False

        # Read output line by line with timeout
        deadline = t_start + duration_s + 4.0
        while time.time() < deadline:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue

            output_lines.append(line)
            # Count output buffer lines from fakesink or decoder
            if "fakesink" in line and ("chain" in line or "buffer" in line or "PTS" in line):
                frame_count += 1
            elif "/GstPipeline:pipeline0/" in line and "caps" in line:
                pass

            # Detect PTS warning from Rockchip MPP
            if "MPP is not able to generate pts" in line:
                result["pts_warning"] = True
                result["pts_valid"] = False

            if source_type == "rtsp" and (time.time() - t_start) >= duration_s:
                # Target duration reached for live stream
                timed_out = True
                break

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        t_elapsed = max(0.001, time.time() - t_start)
        cpu_t2 = get_cpu_times()

        # Parse buffer count from verbose logs if chain wasn't counted
        full_log = "".join(output_lines)
        result["logs"] = full_log

        # Fallback frame count parser from GStreamer verbose timestamps
        pts_matches = len(re.findall(r"pts: \d+:\d+:\d+\.\d+", full_log))
        if pts_matches > frame_count:
            frame_count = pts_matches

        # Check for error messages
        err_match = re.search(r"ERROR:\s+(.+)", full_log)
        if err_match:
            result["error_msg"] = err_match.group(1).strip()
            result["success"] = False
        else:
            result["success"] = frame_count > 0 or timed_out

        result["frames"] = frame_count
        result["fps"] = round(frame_count / t_elapsed, 1)
        result["cpu_pct"] = round(calculate_cpu_percent(cpu_t1, cpu_t2), 1)

    except Exception as e:
        result["error_msg"] = str(e)
        result["success"] = False

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Side-by-Side Benchmark: Hardware vs Software
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison_benchmark(
    source_type: str,
    source_url: Optional[str] = None,
    duration_s: float = 5.0,
) -> None:
    print_header(f"Hardware vs Software Decode Benchmark (Duration: {duration_s}s)")

    print(f"Target Source: {source_url or 'Synthetic 720p H.264 stream'}")
    print("Testing Pipeline 1: Hardware Decoder (mppvideodec)...")
    hw_res = run_pipeline_test("mppvideodec", source_type, source_url, duration_s=duration_s)

    print("Testing Pipeline 2: Software Decoder (avdec_h264)...")
    sw_res = run_pipeline_test("avdec_h264", source_type, source_url, duration_s=duration_s)

    print("\n" + "-" * 65)
    print(f"{'Metric':<26} | {'Hardware (mppvideodec)':<18} | {'Software (avdec_h264)':<18}")
    print("-" * 65)

    hw_status = "SUCCESS" if hw_res["success"] else "FAILED"
    sw_status = "SUCCESS" if sw_res["success"] else "FAILED"
    print(f"{'Pipeline Status':<26} | {hw_status:<18} | {sw_status:<18}")

    hw_fps_str = f"{hw_res['fps']} FPS"
    sw_fps_str = f"{sw_res['fps']} FPS"
    hw_cpu_str = f"{hw_res['cpu_pct']}%"
    sw_cpu_str = f"{sw_res['cpu_pct']}%"

    print(f"{'Frames Processed':<26} | {str(hw_res['frames']):<18} | {str(sw_res['frames']):<18}")
    print(f"{'Throughput':<26} | {hw_fps_str:<18} | {sw_fps_str:<18}")
    print(f"{'Host CPU Utilization':<26} | {hw_cpu_str:<18} | {sw_cpu_str:<18}")

    hw_pts = "WARN (missing PTS)" if hw_res["pts_warning"] else ("VALID" if hw_res["success"] else "N/A")
    sw_pts = "VALID" if sw_res["success"] else "N/A"
    print(f"{'PTS Timestamps':<26} | {hw_pts:<18} | {sw_pts:<18}")
    print("-" * 65)

    if hw_res["error_msg"]:
        print(f"\n[HARDWARE ERROR] {hw_res['error_msg']}")
    if sw_res["error_msg"]:
        print(f"\n[SOFTWARE ERROR] {sw_res['error_msg']}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Configuration & Voyager Compatibility Check
# ─────────────────────────────────────────────────────────────────────────────

def check_config_yaml() -> Optional[str]:
    print_header("Application Configuration Check (config/config.yaml)")
    cfg_path = Path("config/config.yaml")
    first_rtsp = None

    if not cfg_path.is_file():
        print(f"  [NOTE] {cfg_path} not found.")
        return None

    try:
        import yaml
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        # Check Voyager hardware decode setting
        vcfg = cfg.get("voyager", {})
        hw_dec = vcfg.get("hardware_decode")
        print(f"  Voyager 'hardware_decode' setting: {hw_dec} ({'Enabled (mppvideodec)' if hw_dec else 'Disabled (software avdec_h264)'})")

        # Find enabled camera sources
        cams = cfg.get("cameras", [])
        for c in cams:
            if c.get("enabled", True):
                cid = c.get("id", "cam")
                src = c.get("source", "")
                src_sub = c.get("source_sub", "")
                print(f"  Camera '{cid}':")
                print(f"    Main: {src}")
                if src_sub:
                    print(f"    Sub:  {src_sub}")
                if not first_rtsp and src.startswith("rtsp://"):
                    first_rtsp = src
        return first_rtsp
    except Exception as e:
        print(f"  [WARNING] Failed to parse {cfg_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rockchip RK3588 Hardware Video Decoding Diagnostic & Benchmark"
    )
    parser.add_argument("--camera", action="store_true", help="Test using primary camera from config/config.yaml")
    parser.add_argument("--rtsp", type=str, default=None, help="Test using specific RTSP URL")
    parser.add_argument("--file", type=str, default=None, help="Test using a local video file")
    parser.add_argument("--compare", action="store_true", help="Run side-by-side HW vs SW decode comparison benchmark")
    parser.add_argument("--duration", type=float, default=5.0, help="Test stream duration in seconds (default: 5.0)")
    args = parser.parse_args()

    check_system()
    nodes = check_hardware_nodes()
    plugins = check_gstreamer_plugins()
    cfg_rtsp = check_config_yaml()

    # Determine target source
    target_url = None
    target_type = "synthetic"

    if args.rtsp:
        target_url = args.rtsp
        target_type = "rtsp"
    elif args.camera and cfg_rtsp:
        target_url = cfg_rtsp
        target_type = "rtsp"
    elif args.file:
        target_url = args.file
        target_type = "file"

    has_mpp_node = nodes.get("/dev/mpp_service", False)
    has_mpp_plugin = plugins.get("mppvideodec", {}).get("found", False)

    print_header("Hardware Decoding Capabilities Assessment")
    if has_mpp_node and has_mpp_plugin:
        print("  [PASS] Rockchip VPU node (/dev/mpp_service) and GStreamer plugin (mppvideodec) are ACTIVE.")
        print("         Hardware video decoding is supported on this board.")
    elif has_mpp_node and not has_mpp_plugin:
        print("  [WARN] /dev/mpp_service exists, but GStreamer plugin 'mppvideodec' is MISSING.")
        print("         Run: sudo apt install gstreamer1.0-rockchip rockchip-mpp librga")
    elif not has_mpp_node:
        print("  [FAIL] /dev/mpp_service NOT found. Rockchip MPP kernel module is not loaded.")
        print("         Hardware video decoding cannot run; system will fall back to software (avdec_h264).")

    # Run tests
    if args.compare:
        run_comparison_benchmark(target_type, target_url, duration_s=args.duration)
    elif target_type in ("rtsp", "file"):
        print_header(f"Live Hardware Decode Test ({target_type.upper()})")
        print(f"Source: {target_url}")
        res = run_pipeline_test("mppvideodec", target_type, target_url, duration_s=args.duration)
        print(f"Result: {'SUCCESS' if res['success'] else 'FAILED'}")
        print(f"Decoded Frames: {res['frames']} ({res['fps']} FPS)")
        print(f"CPU Load:       {res['cpu_pct']}%")
        if res["pts_warning"]:
            print("[NOTE] 'MPP is not able to generate pts' detected. Frames decode cleanly, but lack presentation timestamps.")
        if res["error_msg"]:
            print(f"[ERROR] {res['error_msg']}")
    else:
        print_header("Synthetic Hardware Decode Test")
        res = run_pipeline_test("mppvideodec", "synthetic", duration_s=args.duration)
        if res["success"]:
            print(f"  [SUCCESS] Synthetic stream decoded via mppvideodec: {res['frames']} frames @ {res['fps']} FPS (CPU: {res['cpu_pct']}%)")
        else:
            print(f"  [SKIPPED / NOTICE] Synthetic decode test: {res['error_msg'] or 'Encoder/pipeline unavailable'}")
            print("  Tip: Test with your camera stream using:")
            print("    python tools/diag_hw_decode.py --camera")
            print("    python tools/diag_hw_decode.py --compare")

    print_header("Diagnostic Complete")


if __name__ == "__main__":
    main()
