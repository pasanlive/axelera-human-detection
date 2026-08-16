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

def load_config(config_path: str) -> dict:
    """Loads system YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found at: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def enroll_face_cli(name: str, image_path: str, config: dict):
    """CLI handler for enrolling a new face into the database."""
    if not os.path.exists(image_path):
        print(f"[ENROLL ERROR] Image file does not exist: {image_path}")
        sys.exit(1)

    print(f"[ENROLLMENT] Processing face enrollment for identity: '{name}'...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ENROLL ERROR] Could not decode image file: {image_path}")
        sys.exit(1)

    db_path = config.get("face_db", {}).get("path", "data/face_db.json")
    face_db = FaceDatabase(db_path=db_path)
    recognizer = FaceRecognizer(config["models"]["face_recognizer"], face_db)

    # Extract embedding from input image
    embedding = recognizer.extract_embedding(img)
    face_db.add_identity(name, embedding)
    print(f"[ENROLLMENT SUCCESS] Identity '{name}' successfully saved in {db_path}.")

def list_faces_cli(config: dict):
    """CLI handler to list enrolled identities."""
    db_path = config.get("face_db", {}).get("path", "data/face_db.json")
    face_db = FaceDatabase(db_path=db_path)
    identities = face_db.list_identities()

    print("\n==========================================================")
    print(f"       Enrolled Identities Gallery ({len(identities)} total)")
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
    no_display = sys.platform.startswith('linux') and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY')
    parser.add_argument("--headless", action="store_true", default=no_display, help="Run without GUI window")
    args = parser.parse_args()

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

    window_name = config.get("visualization", {}).get("window_name", "Axelera Metis Multi-Camera System")

    print("\n[RUNNING] Press 'q' to quit, 's' to save current snapshot frame.\n")

    try:
        while pipeline.is_running:
            start_time = time.time()

            # Execute pipeline step across all active streams
            output_frames = pipeline.process_step()

            # Compose multi-stream output grid
            grid_frame = pipeline.compose_grid(output_frames)

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
