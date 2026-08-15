"""
Visualization: OpenCV Renderer for Human Bboxes, Pose Keypoint Skeletons, and Face Identity Tags.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# COCO 17 Skeletal Connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head / Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms / Shoulders
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

# Color Palettes (BGR)
COLOR_HUMAN_BOX = (255, 178, 50)      # Neon Yellow/Cyan
COLOR_FACE_BOX = (50, 255, 100)       # Bright Green
COLOR_UNKNOWN_FACE = (50, 100, 255)   # Coral Red
COLOR_KEYPOINT = (0, 255, 255)        # Yellow
COLOR_LIMB = (255, 100, 0)            # Deep Blue

class Visualizer:
    """Renders multi-modal detection overlays on video frames."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("visualization", {})
        self.draw_boxes = self.config.get("draw_boxes", True)
        self.draw_pose = self.config.get("draw_pose", True)
        self.draw_faces = self.config.get("draw_faces", True)
        self.draw_fps = self.config.get("draw_fps", True)

    def draw_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]], poses: List[Dict[str, Any]], faces: List[Dict[str, Any]], stream_title: str, fps: float) -> np.ndarray:
        """
        Renders bounding boxes, keypoint skeletons, and face identity tags onto the frame.
        """
        out_frame = frame.copy()

        # 1. Render Human Detections
        if self.draw_boxes:
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                conf = det.get("confidence", 0.0)
                track_id = det.get("track_id", None)

                label = f"Person #{track_id} ({conf:.2f})" if track_id else f"Person ({conf:.2f})"
                
                # Draw bounding box
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), COLOR_HUMAN_BOX, 2)
                
                # Label badge
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out_frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), COLOR_HUMAN_BOX, -1)
                cv2.putText(out_frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Render 17-Keypoint Pose Skeleton
        if self.draw_pose:
            for pose in poses:
                kpts = pose["keypoints"]  # (17, 3) [x, y, conf]
                
                # Draw limb connections
                for k1_idx, k2_idx in SKELETON_CONNECTIONS:
                    if k1_idx < len(kpts) and k2_idx < len(kpts):
                        pt1 = (int(kpts[k1_idx][0]), int(kpts[k1_idx][1]))
                        pt2 = (int(kpts[k2_idx][0]), int(kpts[k2_idx][1]))
                        conf1, conf2 = kpts[k1_idx][2], kpts[k2_idx][2]

                        if conf1 > 0.3 and conf2 > 0.3 and pt1[0] > 0 and pt2[0] > 0:
                            cv2.line(out_frame, pt1, pt2, COLOR_LIMB, 2, cv2.LINE_AA)

                # Draw joint points
                for kx, ky, kc in kpts:
                    if kc > 0.3 and kx > 0 and ky > 0:
                        cv2.circle(out_frame, (int(kx), int(ky)), 4, COLOR_KEYPOINT, -1, cv2.LINE_AA)

        # 3. Render Face Identity Tags
        if self.draw_faces:
            for face in faces:
                x1, y1, x2, y2 = map(int, face["bbox"])
                name = face["name"]
                sim = face.get("similarity", 0.0)

                color = COLOR_FACE_BOX if name != "Unknown" else COLOR_UNKNOWN_FACE
                label = f"{name} ({sim * 100:.0f}%)" if name != "Unknown" else "Unknown Face"

                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out_frame, (x1, y2), (x1 + tw + 6, y2 + th + 6), color, -1)
                cv2.putText(out_frame, label, (x1 + 3, y2 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Render Stream Header & FPS Counter
        if self.draw_fps:
            header_str = f"{stream_title} | FPS: {fps:.1f}"
            cv2.rectangle(out_frame, (10, 10), (10 + len(header_str) * 12 + 10, 42), (20, 20, 20), -1)
            cv2.putText(out_frame, header_str, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        return out_frame
