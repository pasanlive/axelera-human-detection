#!/usr/bin/env python3
"""
Axelera Metis Multi-Camera Detection, Pose, and Face Recognition System Application.

All inference runs on the Metis AIPU through the Voyager SDK (INT8, compiled networks).
Run `python tools/deploy_networks.py` once on the board (or an x86 build host) first.

Usage:
  python app.py                                       # Run real-time multi-camera system
  python app.py --config config/config.yaml          # Run with custom config file
  python app.py --simulate                            # Dashboard/dev mode without Metis hardware
  python app.py --enroll "John Doe" --image face.jpg  # Enroll face identity (runs on the AIPU)
  python app.py --list-faces                          # List enrolled identities
  python app.py --print-plan                          # Show the capacity plan for the configured cameras
"""

import argparse
import json
import os
import signal
import sys
import threading
import time

import yaml

from src.pipeline import MultiCameraPipeline
from src.utils.auto_updater import AutoUpdater
from src.utils.face_db import FaceDatabase
from src.web.server import WebServer


def load_config(config_path: str) -> dict:
    """Loads system YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found at: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def enroll_face_cli(name: str, image_path: str, config: dict, simulate: bool):
    """CLI handler for enrolling a face identity (RetinaFace + FaceNet on the AIPU)."""
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        print(f"[FACE ENROLLMENT FAILED] Cannot read image: {image_path}")
        sys.exit(1)
    pipeline = MultiCameraPipeline(config, simulate=simulate)
    print(f"[FACE ENROLLMENT] Enrolling '{name}' from image: {image_path}...")
    ok, msg = pipeline.enroll_identity(name, image=img)
    pipeline.preview.stop()
    print(f"[FACE ENROLLMENT {'SUCCESS' if ok else 'FAILED'}] {msg}")
    if not ok:
        sys.exit(1)


def list_faces_cli(config: dict):
    """CLI handler for listing all enrolled face identities."""
    db_path = config.get("face_db", {}).get("path", "data/face_db.json")
    face_db = FaceDatabase(db_path=db_path)
    identities = face_db.list_identities()
    print("\n==========================================================")
    print(f" Enrolled Face Identities ({len(identities)})")
    print("==========================================================")
    for idx, name in enumerate(identities, 1):
        print(f" {idx}. {name}")
    print("==========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Axelera Metis Multi-Camera Detection System")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML file")
    parser.add_argument("--simulate", action="store_true", help="Run without Metis hardware (synthetic results)")
    parser.add_argument("--enroll", type=str, help="Enroll face identity name")
    parser.add_argument("--image", type=str, help="Image filepath for face enrollment")
    parser.add_argument("--list-faces", action="store_true", help="List enrolled face identities")
    parser.add_argument("--print-plan", action="store_true", help="Print the capacity plan and exit")
    parser.add_argument("--no-hw-decode", action="store_true", help="Force software video decode (diagnostics)")
    parser.add_argument("--no-tiling", action="store_true", help="Disable tiled (ultra) inference (diagnostics)")
    parser.add_argument("--debug", action="store_true", help="Enable extensive diagnostic logging (GStreamer, Level-Zero, and hardware watchdog)")
    parser.add_argument("--no-tracers", action="store_true", help="Disable background axtrace processes (prevents concurrent /dev/metis* access)")
    parser.add_argument("--aipu-clock", type=str, default=None, help="Override AIPU clock in MHz (e.g. 400, 800, or none)")
    parser.add_argument("--mvm-limit", type=str, default=None, help="Override MVM utilization limit percentage (e.g. 40, 50, 100, or none)")
    parser.add_argument("--aipu-cores", type=int, default=None, help="Override number of AIPU cores (1, 2, or 4)")
    parser.add_argument("--headless", action="store_true", help="Run in strict headless mode (unsets DISPLAY and disables GUI probes)")
    parser.add_argument("--web", action="store_true", default=False, help="Enable local network web dashboard interface")
    parser.add_argument("--no-web", action="store_true", help="Disable web dashboard interface")
    parser.add_argument("--host", type=str, default=None, help="Web server host IP address")
    parser.add_argument("--port", type=int, default=None, help="Web server port")
    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS mode and run plain HTTP server")
    parser.add_argument("--cert", type=str, default=None, help="SSL certificate filepath")
    parser.add_argument("--key", type=str, default=None, help="SSL private key filepath")
    parser.add_argument("--reload-firmware", action="store_true", help="Force reload Metis AIPU firmware before startup")
    args = parser.parse_args()

    # Enable crash handler for C++ abort / L0::Exception
    from src.utils.hw_diag import enable_crash_handler, get_watchdog
    enable_crash_handler()

    if args.reload_firmware:
        get_watchdog().ensure_device_ready(force=True)

    if args.debug:
        os.environ["ZE_DEBUG"] = "1"
        os.environ["AXELERA_LOG_LEVEL"] = "DEBUG"
        if not os.environ.get("GST_DEBUG"):
            os.environ["GST_DEBUG"] = "2,openclallocator:0,axinplace:2,axinferencenet:4"
        print("[DIAG] Detailed diagnostic logging active (ZE_DEBUG=1, GST_DEBUG=2,openclallocator:0,axinplace:2,axinferencenet:4)")

    # Optimize Voyager runtime environment for Rockchip RK3588 + Metis PCIe:
    # 1. Disable OpenCL preprocessing on host GPU (ARM Mali G610) to avoid memory queue conflicts
    # 2. Disable double buffering to prevent command list build-up and -1301 wait kernel timeouts
    # 3. Disable zero-copy dmabuf to prevent CMA heap fragmentation with software video decoding
    # 4. Enforce TCP RTSP transport for lossless frame delivery without dropped reference frames
    os.environ.setdefault("AX_DISABLE_OPENCL", "1")
    os.environ.setdefault("AX_OPENCL_PREFERENCE", "NONE")
    os.environ.setdefault("AXELERA_USE_CL_DOUBLE_BUFFER", "0")
    os.environ.setdefault("AXELERA_USE_DOUBLE_BUFFER", "0")
    os.environ.setdefault("AXELERA_USE_DMABUF", "0")
    os.environ.setdefault("AXELERA_RTSP_PROTOCOL", "tcp")

    if args.headless or not os.environ.get("DISPLAY"):
        # Explicit headless mode: prevent X11 auth warnings and Pyglet display probe errors
        os.environ.pop("DISPLAY", None)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["MPLBACKEND"] = "Agg"

    config = load_config(args.config)
    # Runtime-only overrides (not persisted unless the dashboard later saves config).
    if args.no_hw_decode:
        config.setdefault("voyager", {})["hardware_decode"] = False
        print("[DIAG] Hardware video decode disabled (software decodebin)")
    if args.no_tiling:
        config.setdefault("capacity", {})["enable_tiling"] = False
        print("[DIAG] Tiled inference disabled")
    if args.no_tracers:
        config.setdefault("voyager", {})["disable_tracers"] = True
        print("[DIAG] Background axtrace tracers disabled")
    if args.aipu_clock is not None:
        val = None if args.aipu_clock.lower() in ("none", "0") else int(args.aipu_clock)
        config.setdefault("voyager", {})["aipu_clock_mhz"] = val
        print(f"[DIAG] AIPU clock profile override: {val}")
    if args.mvm_limit is not None:
        val = None if args.mvm_limit.lower() in ("none", "0") else int(args.mvm_limit)
        config.setdefault("voyager", {})["mvm_utilization"] = val
        print(f"[DIAG] MVM utilization override: {val}")
    if args.aipu_cores is not None:
        config.setdefault("voyager", {})["aipu_cores"] = args.aipu_cores
        print(f"[DIAG] AIPU cores override: {args.aipu_cores}")
    if args.headless:
        print("[DIAG] Strict headless mode active (DISPLAY disabled)")

    if args.enroll:
        if not args.image:
            print("[ERROR] Please provide --image filepath for face enrollment.")
            sys.exit(1)
        enroll_face_cli(args.enroll, args.image, config, args.simulate)
        return

    if args.list_faces:
        list_faces_cli(config)
        return

    # Clean up any orphaned axtrace processes from previous aborted runs before initializing Metis
    if not (args.simulate or args.print_plan):
        try:
            import subprocess
            subprocess.run(["pkill", "-9", "axtrace"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # Planning is pure Python, so --print-plan never needs the SDK/hardware.
    pipeline = MultiCameraPipeline(config, simulate=args.simulate or args.print_plan, config_path=args.config)

    if args.print_plan:
        print(json.dumps(pipeline.get_capacity_plan(), indent=2))
        pipeline.preview.stop()
        return

    shutdown = threading.Event()

    # Application restart callback for Auto-Updater
    def restart_application():
        print("[AUTO-UPDATER] Gracefully stopping pipeline and restarting application...")
        try:
            pipeline.stop()
        except Exception:
            pass
        time.sleep(1.0)
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    auto_updater = AutoUpdater(config, config_path=args.config, restart_callback=restart_application)

    # Dashboard first: building the Voyager stream (model load, GStreamer preroll, RTSP
    # connect) can take a while or hang, and the dashboard must stay reachable meanwhile.
    web_cfg = config.get("web", {})
    web_enabled = (web_cfg.get("enabled", True) and not args.no_web) or args.web
    if web_enabled:
        use_https = False if args.no_https else web_cfg.get("https", True)
        web_server = WebServer(
            pipeline,
            host=args.host or web_cfg.get("host", "0.0.0.0"),
            port=args.port or int(web_cfg.get("port", 8000)),
            use_https=use_https,
            cert_file=args.cert or web_cfg.get("cert_file", "data/ssl/cert.pem"),
            key_file=args.key or web_cfg.get("key_file", "data/ssl/key.pem"),
            auto_updater=auto_updater,
        )
        web_server.start_background()

    start_state: dict = {"done": False, "error": None, "t0": time.time()}

    def _start_pipeline():
        try:
            pipeline.start()
        except BaseException as e:  # noqa: BLE001 - reported by the main loop
            start_state["error"] = e
        finally:
            start_state["done"] = True

    threading.Thread(target=_start_pipeline, name="pipeline-start", daemon=True).start()

    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    print("\n[STARTING] Building Voyager inference stream in the background. Press Ctrl+C to stop.\n")
    announced = False
    try:
        while not shutdown.is_set():
            shutdown.wait(5.0)
            err = start_state["error"]
            if err is not None:
                if err.__class__.__name__ == "NetworkNotCompiledError":
                    print(err, file=sys.stderr)
                else:
                    import traceback
                    traceback.print_exception(type(err), err, err.__traceback__)
                    print("[FATAL] Voyager stream failed to start (see above).", file=sys.stderr)
                os._exit(1)
            if not start_state["done"]:
                waited = int(time.time() - start_state["t0"])
                print(f"[STARTING] Still building Voyager stream ({waited}s) - loading models / "
                      f"GStreamer preroll / RTSP connect...")
                if waited >= 60 and waited % 30 < 5:
                    print("[STARTING] >60s usually means the RTSP source is unreachable or the decoder "
                          "is stuck. Check the camera URL, try --no-hw-decode, or run with GST_DEBUG=3.")
                continue
            if not announced:
                print("\n[RUNNING] Inference is running on the Metis AIPU.\n")
                announced = True
            stats = pipeline.get_stats()
            plan = pipeline.plan
            if plan is not None:
                cams = pipeline.get_camera_status() if hasattr(pipeline, "get_camera_status") else []
                waiting = [c["id"] for c in cams if c.get("enabled") and c.get("state") not in ("live",)]
                face_det_on = pipeline.is_model_enabled("face_detector") if hasattr(pipeline, "is_model_enabled") else False
                face_rec_on = pipeline.is_model_enabled("face_recognizer") if hasattr(pipeline, "is_model_enabled") else False
                lpr_on = pipeline.is_model_enabled("license_plate_recognizer") if hasattr(pipeline, "is_model_enabled") else False

                face_str = ""
                if face_rec_on:
                    face_str = f" known_faces={stats.get('faces_recognized', 0)}"
                elif face_det_on:
                    face_str = f" faces={stats.get('faces_detected', 0)}"

                plate_str = f" plates={stats.get('plates', 0)}" if lpr_on else ""

                rt_metrics = pipeline.get_runtime_metrics() if hasattr(pipeline, "get_runtime_metrics") else {}
                wd_metrics = rt_metrics.get("watchdog", {})
                lat_str = ""
                frames_str = ""
                if wd_metrics:
                    lat_str = f" lat={wd_metrics.get('last_latency_ms', 0):.1f}ms (p95={wd_metrics.get('p95_latency_ms', 0):.1f}ms)"
                    frames_str = f" frames={wd_metrics.get('frames_processed', 0)}"

                core_temp = rt_metrics.get("core_temp", {}).get("value")
                temp_str = f" temp={core_temp:.0f}°C" if core_temp is not None else ""

                print(f"[STATUS] tier={plan.tier.name} target={plan.fps}fps/cam achieved={stats['total_fps']}fps total "
                      f"persons={stats['persons']}{face_str}{plate_str}{frames_str}{lat_str}{temp_str}"
                      + (f" waiting_for_frames={waiting}" if waiting else ""))
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping pipeline...")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[FATAL ERROR] Main loop encountered unhandled exception: {e}", file=sys.stderr)
    finally:
        auto_updater.stop()
        pipeline.stop()
        try:
            import subprocess
            subprocess.run(["pkill", "-9", "axtrace"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        print("[SHUTDOWN] Axelera Metis system stopped cleanly.")
        os._exit(0)


if __name__ == "__main__":
    main()
