"""
StreamManager: Concurrent Multi-Camera Video Stream Handler.
"""

import threading
import time
from typing import Dict, Optional, Tuple, List
import numpy as np

from src.camera.base_camera import BaseCamera, OpenCVCamera, SyntheticCamera

class StreamManager:
    """Manages multiple camera feeds in dedicated background threads."""

    def __init__(self, camera_configs: List[dict]):
        self.camera_configs = camera_configs
        self.cameras: Dict[str, BaseCamera] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.latest_frames: Dict[str, Optional[np.ndarray]] = {}
        self.lock = threading.Lock()
        self.is_running = False

    def initialize_cameras(self):
        """Initializes camera instances from config."""
        for cfg in self.camera_configs:
            if not cfg.get("enabled", True):
                continue

            cam_id = cfg.get("id", f"cam_{len(self.cameras)}")
            name = cfg.get("name", cam_id)
            source = str(cfg.get("source", "0"))
            fps_limit = cfg.get("fps_limit", 30)

            # Instantiation logic
            if source.lower() == "synthetic" or source == "-1":
                cam = SyntheticCamera(cam_id, name, source, fps_limit)
                cam.open()
            else:
                cam = OpenCVCamera(cam_id, name, source, fps_limit)
                if not cam.open():
                    print(f"[STREAM MANAGER] Camera {cam_id} ({source}) unavailable. Falling back to Synthetic Camera.")
                    cam = SyntheticCamera(cam_id, name, "synthetic", fps_limit)
                    cam.open()

            self.cameras[cam_id] = cam
            self.latest_frames[cam_id] = None

    def start(self):
        """Starts background frame reader threads for all cameras."""
        self.is_running = True
        for cam_id, cam in self.cameras.items():
            t = threading.Thread(target=self._capture_loop, args=(cam_id, cam), daemon=True)
            self.threads[cam_id] = t
            t.start()
        print(f"[STREAM MANAGER] Started {len(self.threads)} camera background capture threads.")

    def _capture_loop(self, cam_id: str, cam: BaseCamera):
        """Background thread loop per camera."""
        while self.is_running and cam.is_running:
            ret, frame = cam.read_frame()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frames[cam_id] = frame
            else:
                time.sleep(0.01)

    def get_latest_frame(self, cam_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Thread-safe retrieval of latest frame from camera_id."""
        with self.lock:
            frame = self.latest_frames.get(cam_id, None)
            if frame is not None:
                return True, frame.copy()
            return False, None

    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """Returns a snapshot dictionary of latest frames for all active cameras."""
        frames = {}
        with self.lock:
            for cam_id, frame in self.latest_frames.items():
                if frame is not None:
                    frames[cam_id] = frame.copy()
        return frames

    def stop(self):
        """Stops all threads and releases camera resources."""
        print("[STREAM MANAGER] Stopping camera feeds...")
        self.is_running = False
        for cam in self.cameras.values():
            cam.close()
        for t in self.threads.values():
            t.join(timeout=1.0)
        print("[STREAM MANAGER] All camera feeds stopped.")
