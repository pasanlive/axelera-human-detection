#!/usr/bin/env python3
"""
Axelera Metis Multi-Camera Detection, Pose, and Face Recognition System Application.

Usage:
  python app.py                                       # Run real-time multi-camera system
  python app.py --config config/config.yaml          # Run with custom config file
  python app.py --enroll "John Doe" --image face.jpg  # Enroll face identity
  python app.py --list-faces                          # List enrolled identities
  python app.py --headless                            # Run in headless mode (no GUI window)
"""

import sys
import os
import time
import argparse
import yaml
import cv2

from src.pipeline import MultiCameraPipeline
from src.utils.face_db import FaceDatabase
from src.inference.face_recognizer import FaceRecognizer
from src.web.server import WebServer

def load_config(config_path: str) -> dict:
    """Loads system YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found at: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def enroll_face_cli(name: str, image_path: str, config: dict):
    """CLI handler for enrolling a face identity."""
    db_path = config.get("face_db", {}).get("path", "data/face_db.json")
    face_db = FaceDatabase(db_path=db_path)
    recognizer = FaceRecognizer(config["models"]["face_recognizer"], face_db)

    print(f"[FACE ENROLLMENT] Enrolling '{name}' from image: {image_path}...")
    success = recognizer.enroll_identity(name, image_path)
    if success:
        print(f"[FACE ENROLLMENT SUCCESS] Identity '{name}' successfully enrolled.")
    else:
        print(f"[FACE ENROLLMENT FAILED] Could not detect a valid face in: {image_path}")

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
    parser.add_argument("--enroll", type=str, help="Enroll face identity name")
    parser.add_argument("--image", type=str, help="Image filepath for face enrollment")
    parser.add_argument("--list-faces", action="store_true", help="List enrolled face identities")
    parser.add_argument("--gui", action="store_true", help="Enable desktop OpenCV window display (default: False)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without GUI window")
    parser.add_argument("--web", action="store_true", default=True, help="Enable local network web dashboard interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--https", action="store_true", default=True, help="Enable HTTPS mode with self-signed SSL cert")
    parser.add_argument("--cert", type=str, default="data/ssl/cert.pem", help="SSL certificate filepath")
    parser.add_argument("--key", type=str, default="data/ssl/key.pem", help="SSL private key filepath")
    args = parser.parse_args()

    # Headless mode is default unless --gui is explicitly requested
    if args.gui:
        args.headless = False
    else:
        args.headless = True

    config = load_config(args.config)

    # Handle enrollment CLI subcommand
    if args.enroll:
        if not args.image:
            print("[ERROR] Please provide --image filepath for face enrollment.")
            sys.exit(1)
        enroll_face_cli(args.enroll, args.image, config)
        return

    # Handle list faces CLI subcommand
    if args.list_faces:
        list_faces_cli(config)
        return

    # Start main multi-camera pipeline
    pipeline = MultiCameraPipeline(config)
    pipeline.start()

    # Start Remote Web Dashboard Server
    web_server = None
    if args.web:
        web_cfg = config.get("web", {})
        use_https = args.https if args.https is not None else web_cfg.get("https", True)
        cert_file = args.cert or web_cfg.get("cert_file", "data/ssl/cert.pem")
        key_file = args.key or web_cfg.get("key_file", "data/ssl/key.pem")
        web_server = WebServer(pipeline, host=args.host, port=args.port, use_https=use_https, cert_file=cert_file, key_file=key_file)
        web_server.start_background()

    window_name = config.get("visualization", {}).get("window_name", "Axelera Metis Multi-Camera System")

    print("\n[RUNNING] Press 'q' to quit, 's' to save current snapshot frame.\n")

    try:
        while pipeline.is_running:
            try:
                start_time = time.time()

                # Execute pipeline step across all active streams
                output_frames = pipeline.process_step()

                # Compose multi-stream output grid
                grid_frame = pipeline.compose_grid(output_frames)

                # Broadcast frame buffers to Web Server
                if web_server:
                    web_server.update_frame(grid_frame, output_frames)

                if not args.headless:
                    try:
                        cv2.imshow(window_name, grid_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            print("[EXIT] User requested shutdown.")
                            break
                        elif key == ord('s'):
                            snapshot_filename = f"snapshot_{int(time.time())}.jpg"
                            cv2.imwrite(snapshot_filename, grid_frame)
                            print(f"[SAVED] Saved frame snapshot to: {snapshot_filename}")
                    except Exception as e:
                        print(f"[HEADLESS AUTO-SWITCH] GUI display unavailable ({e}). Continuing in headless mode...")
                        args.headless = True

                # Sleep slightly to maintain clean CPU iteration rate
                elapsed = time.time() - start_time
                sleep_time = max(0.001, (1.0 / 60.0) - elapsed)
                time.sleep(sleep_time)

            except Exception as step_err:
                print(f"[PIPELINE RUNTIME WARNING] Step execution error: {step_err}")
                import traceback
                traceback.print_exc()
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping pipeline...")
    finally:
        pipeline.stop()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[SHUTDOWN] Axelera Metis system stopped cleanly.")
        os._exit(0)

if __name__ == "__main__":
    main()
