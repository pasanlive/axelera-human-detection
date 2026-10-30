"""
On-demand preview rendering.

Inference results arrive for every camera at the planned FPS, but nobody needs a
rendered JPEG for a camera that is not on screen. ``PreviewHub`` therefore:

* tracks active viewers per camera (and for the grid) - MJPEG generators
  acquire/release them;
* asks the pipeline for a frame only when a viewer exists and the preview
  interval has elapsed (``wants_frame``), so frame download from the Voyager
  buffer, overlay drawing, resizing and JPEG encoding happen at preview FPS for
  viewed cameras only;
* renders on its own worker thread so the inference consumer never blocks.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

GRID = "grid"


class PreviewHub:
    def __init__(self, renderer: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
                 preview_fps: float = 12.0, grid_fps: float = 6.0, max_width: int = 1280,
                 grid_size: Tuple[int, int] = (1280, 720), jpeg_quality: int = 75):
        self.renderer = renderer
        self.preview_fps = max(0.5, float(preview_fps))
        self.grid_fps = max(0.5, float(grid_fps))
        self.max_width = int(max_width)
        self.grid_size = grid_size
        self.jpeg_quality = int(jpeg_quality)

        self._viewers: Counter = Counter()
        self._last_capture: Dict[str, float] = {}
        self._pending: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self._tiles: Dict[str, np.ndarray] = {}  # latest rendered (downscaled) frame per camera
        self._jpegs: Dict[str, Tuple[int, bytes]] = {}  # key -> (seq, jpeg)
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._order: list = []  # camera display order for the grid
        self._thread = threading.Thread(target=self._loop, name="preview-render", daemon=True)
        self._thread.start()

    # ── viewer bookkeeping ──────────────────────────────────────────────────
    def acquire(self, key: str) -> None:
        with self._cond:
            self._viewers[key] += 1

    def release(self, key: str) -> None:
        with self._cond:
            self._viewers[key] -= 1
            if self._viewers[key] <= 0:
                del self._viewers[key]

    def viewer_count(self, key: Optional[str] = None) -> int:
        with self._cond:
            return sum(self._viewers.values()) if key is None else self._viewers.get(key, 0)

    def set_camera_order(self, cam_ids) -> None:
        with self._cond:
            self._order = list(cam_ids)
            for k in [k for k in self._tiles if k not in self._order]:
                self._tiles.pop(k, None)
                self._jpegs.pop(k, None)

    # ── producer side (inference consumer thread) ───────────────────────────
    def wants_frame(self, cam_id: str, now: Optional[float] = None) -> bool:
        now = now or time.time()
        with self._cond:
            direct = self._viewers.get(cam_id, 0) > 0
            grid = self._viewers.get(GRID, 0) > 0
            if not (direct or grid):
                return False
            interval = 1.0 / (self.preview_fps if direct else self.grid_fps)
            if now - self._last_capture.get(cam_id, 0.0) < interval:
                return False
            self._last_capture[cam_id] = now
            return True

    def submit(self, cam_id: str, frame: np.ndarray, packet: Dict[str, Any]) -> None:
        with self._cond:
            self._pending[cam_id] = (frame, packet)  # newest wins
            self._cond.notify_all()

    # ── consumer side (HTTP handlers) ───────────────────────────────────────
    def latest(self, key: str) -> Optional[Tuple[int, bytes]]:
        with self._cond:
            return self._jpegs.get(key)

    def wait_next(self, key: str, last_seq: int, timeout: float = 1.0) -> Optional[Tuple[int, bytes]]:
        deadline = time.time() + timeout
        with self._cond:
            while not self._stop.is_set():
                item = self._jpegs.get(key)
                if item is not None and item[0] != last_seq:
                    return item
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
        return None

    def snapshot(self, key: str = GRID, timeout: float = 2.0) -> Optional[bytes]:
        """Returns a fresh JPEG even if nobody is currently watching."""
        item = self.latest(key)
        if item is not None:
            return item[1]
        self.acquire(key)
        try:
            item = self.wait_next(key, -1, timeout)
            return item[1] if item else None
        finally:
            self.release(key)

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    # ── worker ──────────────────────────────────────────────────────────────
    def _encode(self, img: np.ndarray) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return buf.tobytes() if ok else None

    def _publish(self, key: str, jpeg: Optional[bytes]) -> None:
        if jpeg is None:
            return
        with self._cond:
            self._seq += 1
            self._jpegs[key] = (self._seq, jpeg)
            self._cond.notify_all()

    def _loop(self) -> None:
        last_grid = 0.0
        while not self._stop.is_set():
            with self._cond:
                if not self._pending:
                    self._cond.wait(timeout=1.0 / self.grid_fps)
                pending, self._pending = self._pending, {}
                direct = {k for k, v in self._viewers.items() if v > 0}
            for cam_id, (frame, packet) in pending.items():
                try:
                    rendered = self.renderer(frame, packet)
                    h, w = rendered.shape[:2]
                    if w > self.max_width:
                        rendered = cv2.resize(rendered, (self.max_width, int(h * self.max_width / w)),
                                              interpolation=cv2.INTER_AREA)
                    with self._cond:
                        self._tiles[cam_id] = rendered
                    if cam_id in direct:
                        self._publish(cam_id, self._encode(rendered))
                except Exception as e:  # never kill the render thread
                    print(f"[PREVIEW] render error for {cam_id}: {e}")
            now = time.time()
            if GRID in direct and now - last_grid >= 1.0 / self.grid_fps:
                last_grid = now
                try:
                    self._publish(GRID, self._encode(self.compose_grid()))
                except Exception as e:
                    print(f"[PREVIEW] compose_grid error: {e}")

    def compose_grid(self) -> np.ndarray:
        gw, gh = self.grid_size
        with self._cond:
            order = [c for c in self._order if c in self._tiles] or list(self._tiles)
            tiles = [self._tiles[c] for c in order]
        if not tiles:
            blank = np.zeros((gh, gw, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera streams...", (gw // 2 - 220, gh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)
            return blank
        # Determine reference aspect ratio from active tiles (defaulting to standard 16:9)
        aspect_ratio = 16.0 / 9.0
        for t in tiles:
            if t is not None and hasattr(t, "shape") and t.shape[0] > 0 and t.shape[1] > 0:
                aspect_ratio = float(t.shape[1]) / float(t.shape[0])
                break

        n = len(tiles)
        cols = 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4
        rows = int(np.ceil(n / cols))

        tw = gw // cols
        th = max(1, int(round(tw / aspect_ratio)))
        grid = np.zeros((th * rows, tw * cols, 3), dtype=np.uint8)

        for i, t in enumerate(tiles):
            if t is None or not hasattr(t, "shape") or t.shape[0] == 0 or t.shape[1] == 0:
                continue
            if t.ndim == 2:
                t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            elif t.ndim == 3 and t.shape[2] == 1:
                t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

            ih, iw = t.shape[:2]
            scale = min(tw / iw, th / ih)
            nw = max(1, int(round(iw * scale)))
            nh = max(1, int(round(ih * scale)))
            resized = cv2.resize(t, (nw, nh), interpolation=cv2.INTER_AREA)

            r, c = divmod(i, cols)
            ox = c * tw + (tw - nw) // 2
            oy = r * th + (th - nh) // 2
            grid[oy:oy + nh, ox:ox + nw] = resized

        return grid
