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
    parser.add_argument("--web", action="store_true", default=False, help="Enable local network web dashboard interface")
    parser.add_argument("--no-web", action="store_true", help="Disable web dashboard interface")
    parser.add_argument("--host", type=str, default=None, help="Web server host IP address")
    parser.add_argument("--port", type=int, default=None, help="Web server port")
    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS mode and run plain HTTP server")
    parser.add_argument("--cert", type=str, default=None, help="SSL certificate filepath")
    parser.add_argument("--key", type=str, default=None, help="SSL private key filepath")
    args = parser.parse_args()

    config = load_config(args.config)
    # Runtime-only overrides (not persisted unless the dashboard later saves config).
    if args.no_hw_decode:
        config.setdefault("voyager", {})["hardware_decode"] = False
        print("[DIAG] Hardware video decode disabled (software decodebin)")
    if args.no_tiling:
        config.setdefault("capacity", {})["enable_tiling"] = False
        print("[DIAG] Tiled inference disabled")

    if args.enroll:
        if not args.image:
            print("[ERROR] Please provide --image filepath for face enrollment.")
            sys.exit(1)
        enroll_face_cli(args.enroll, args.image, config, args.simulate)
        return

    if args.list_faces:
        list_faces_cli(config)
        return

    # Planning is pure Python, so --print-plan never needs the SDK/hardware.
    pipeline = MultiCameraPipeline(config, simulate=args.simulate or args.print_plan, config_path=args.config)

    if args.print_plan:
        print(json.dumps(pipeline.get_capacity_plan(), indent=2))
        pipeline.preview.stop()
        return

    try:
        pipeline.start()
    except Exception as e:
        if e.__class__.__name__ == "NetworkNotCompiledError":
            print(e, file=sys.stderr)
            pipeline.preview.stop()
            sys.exit(1)
        raise

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

    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    print("\n[RUNNING] Inference is running on the Metis AIPU. Press Ctrl+C to stop.\n")
    try:
        while not shutdown.is_set():
            shutdown.wait(5.0)
            stats = pipeline.get_stats()
            plan = pipeline.plan
            if plan is not None:
                print(f"[STATUS] tier={plan.tier.name} target={plan.fps}fps/cam achieved={stats['total_fps']}fps total "
                      f"persons={stats['persons']} known_faces={stats['faces_recognized']}")
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping pipeline...")
    finally:
        auto_updater.stop()
        pipeline.stop()
        print("[SHUTDOWN] Axelera Metis system stopped cleanly.")
        os._exit(0)


if __name__ == "__main__":
    main()
