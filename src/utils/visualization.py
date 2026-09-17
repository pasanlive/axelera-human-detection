"""
Visualization: OpenCV Renderer for Human Bboxes, Pose Keypoint Skeletons,
Face Identity Tags, License Plate Text, and Zone Crossing Overlays.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# COCO 17 Skeletal Connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head / Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms / Shoulders
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

# Color Palettes (BGR)
COLOR_HUMAN_BOX   = (255, 178, 50)    # Neon Yellow/Cyan
COLOR_FACE_BOX    = (50, 255, 100)    # Bright Green
COLOR_UNKNOWN_FACE= (50, 100, 255)    # Coral Red
COLOR_KEYPOINT    = (0, 255, 255)     # Yellow
COLOR_LIMB        = (255, 100, 0)     # Deep Blue
COLOR_PLATE_BOX   = (0, 220, 255)     # Amber/Orange
COLOR_PLATE_TEXT  = (0, 0, 0)         # Black text on plate badge
COLOR_ZONE_LINE   = (0, 255, 180)     # Cyan-green
COLOR_ZONE_POLY   = (255, 50, 150)    # Magenta
COLOR_ZONE_TEXT   = (255, 255, 255)   # White


class Visualizer:
    """Renders multi-modal detection overlays on video frames."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("visualization", {})
        self.draw_boxes  = self.config.get("draw_boxes", True)
        self.draw_pose   = self.config.get("draw_pose", True)
        self.draw_faces  = self.config.get("draw_faces", True)
        self.draw_fps    = self.config.get("draw_fps", True)
        self.draw_plates = self.config.get("draw_plates", True)
        self.draw_zones  = self.config.get("draw_zones", True)

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        poses: List[Dict[str, Any]],
        faces: List[Dict[str, Any]],
        stream_title: str,
        fps: float,
        plates: Optional[List[Dict[str, Any]]] = None,
        zones: Optional[List[Dict[str, Any]]] = None,
        zone_counts: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Renders all detection overlays on a frame.

        :param frame:        BGR source frame
        :param detections:   Human bounding boxes [{bbox, confidence, track_id}]
        :param poses:        Pose keypoints [{bbox, confidence, keypoints (17,3)}]
        :param faces:        Face identities [{bbox, name, similarity}]
        :param stream_title: Stream label shown in header
        :param fps:          Current stream FPS
        :param plates:       License plate detections [{bbox, plate_text, confidence, ocr_confidence}]
        :param zones:        Zone definitions for this camera [{id, name, type, points, color}]
        :param zone_counts:  Crossing counts per zone {zone_id -> {direction -> count}}
        :return:             Annotated BGR frame
        """
        out_frame = frame.copy()

        # 1. Zone Overlays (draw behind detections for proper layering)
        if self.draw_zones and zones:
            out_frame = self._draw_zones(out_frame, zones, zone_counts or {})

        # 2. Human Detection Bounding Boxes
        if self.draw_boxes:
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                conf = det.get("confidence", 0.0)
                track_id = det.get("track_id", None)
                label = f"Person #{track_id} ({conf:.2f})" if track_id else f"Person ({conf:.2f})"

                cv2.rectangle(out_frame, (x1, y1), (x2, y2), COLOR_HUMAN_BOX, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out_frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), COLOR_HUMAN_BOX, -1)
                cv2.putText(out_frame, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. 17-Keypoint Pose Skeleton
        if self.draw_pose:
            for pose in poses:
                kpts = pose["keypoints"]  # (17, 3) [x, y, conf]
                for k1_idx, k2_idx in SKELETON_CONNECTIONS:
                    if k1_idx < len(kpts) and k2_idx < len(kpts):
                        pt1 = (int(kpts[k1_idx][0]), int(kpts[k1_idx][1]))
                        pt2 = (int(kpts[k2_idx][0]), int(kpts[k2_idx][1]))
                        conf1, conf2 = kpts[k1_idx][2], kpts[k2_idx][2]
                        if conf1 > 0.3 and conf2 > 0.3 and pt1[0] > 0 and pt2[0] > 0:
                            cv2.line(out_frame, pt1, pt2, COLOR_LIMB, 2, cv2.LINE_AA)
                for kx, ky, kc in kpts:
                    if kc > 0.3 and kx > 0 and ky > 0:
                        cv2.circle(out_frame, (int(kx), int(ky)), 4, COLOR_KEYPOINT, -1, cv2.LINE_AA)

        # 4. Face Identity Tags
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
                cv2.putText(out_frame, label, (x1 + 3, y2 + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 5. License Plate Annotations
        if self.draw_plates and plates:
            for plate in plates:
                x1, y1, x2, y2 = map(int, plate["bbox"])
                text = plate.get("plate_text", "")
                conf = plate.get("confidence", 0.0)
                ocr_conf = plate.get("ocr_confidence", 0.0)

                cv2.rectangle(out_frame, (x1, y1), (x2, y2), COLOR_PLATE_BOX, 2)
                if text:
                    plate_label = f"{text} ({ocr_conf * 100:.0f}%)"
                else:
                    plate_label = f"Plate ({conf:.2f})"

                (tw, th), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                badge_y = y1 - th - 8
                if badge_y < 0:
                    badge_y = y2 + 4
                cv2.rectangle(out_frame, (x1, badge_y), (x1 + tw + 8, badge_y + th + 6),
                              COLOR_PLATE_BOX, -1)
                cv2.putText(out_frame, plate_label, (x1 + 4, badge_y + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PLATE_TEXT, 2, cv2.LINE_AA)

        # 6. Stream Header & FPS Counter
        if self.draw_fps:
            header_str = f"{stream_title} | FPS: {fps:.1f}"
            header_w = len(header_str) * 12 + 10
            cv2.rectangle(out_frame, (10, 10), (10 + header_w, 42), (20, 20, 20), -1)
            cv2.putText(out_frame, header_str, (18, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        return out_frame

    def _draw_zones(self, frame: np.ndarray,
                    zones: List[Dict[str, Any]],
                    zone_counts: Dict[str, Any]) -> np.ndarray:
        """
        Draws zone overlays: dashed lines for line zones, semi-transparent fills for polygons.
        Shows crossing count badges per zone.
        """
        overlay = frame.copy()

        for zone in zones:
            zone_id = zone.get("id", "")
            zone_type = zone.get("type", "line")
            zone_name = zone.get("name", zone_id)
            pts = zone.get("points", [])
            color = tuple(int(c) for c in zone.get("color", [0, 255, 180]))

            if len(pts) < 2:
                continue

            pts_arr = np.array(pts, dtype=np.int32)

            if zone_type == "line" and len(pts) >= 2:
                # Draw dashed line
                p1 = tuple(pts[0])
                p2 = tuple(pts[1])
                self._draw_dashed_line(overlay, p1, p2, color, thickness=2, dash_len=20, gap_len=10)

                # Zone name label at midpoint
                mid_x = (pts[0][0] + pts[1][0]) // 2
                mid_y = (pts[0][1] + pts[1][1]) // 2

                counts = zone_counts.get(zone_id, {}).get("counts", {})
                direction_labels = zone.get("direction_labels", ["IN", "OUT"])
                count_str = "  ".join(f"{d}:{counts.get(d, 0)}" for d in direction_labels)
                label = f"  {zone_name}: {count_str}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (mid_x - 4, mid_y - th - 8),
                              (mid_x + tw + 4, mid_y + 4), (20, 20, 20), -1)
                cv2.putText(overlay, label, (mid_x, mid_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            elif zone_type == "polygon" and len(pts) >= 3:
                # Semi-transparent polygon fill
                poly_overlay = overlay.copy()
                # Softer fill color (lower alpha blend)
                fill_color = tuple(int(c * 0.4) for c in color)
                cv2.fillPoly(poly_overlay, [pts_arr], fill_color)
                cv2.addWeighted(poly_overlay, 0.35, overlay, 0.65, 0, overlay)

                # Polygon border
                cv2.polylines(overlay, [pts_arr], isClosed=True, color=color, thickness=2)

                # Zone name label near centroid
                centroid = pts_arr.mean(axis=0).astype(int)
                counts = zone_counts.get(zone_id, {}).get("counts", {})
                direction_labels = zone.get("direction_labels", ["ENTER", "EXIT"])
                count_str = "  ".join(f"{d}:{counts.get(d, 0)}" for d in direction_labels)
                label = f"{zone_name}  {count_str}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lx = max(0, centroid[0] - tw // 2)
                ly = max(th + 6, centroid[1])
                cv2.rectangle(overlay, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 2), (20, 20, 20), -1)
                cv2.putText(overlay, label, (lx, ly - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return overlay

    @staticmethod
    def _draw_dashed_line(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int = 2,
                          dash_len: int = 20, gap_len: int = 10):
        """Draws a dashed line segment between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        total_len = np.hypot(dx, dy)
        if total_len < 1:
            return
        ux = dx / total_len
        uy = dy / total_len
        step = dash_len + gap_len
        num_steps = int(total_len / step) + 1

        for i in range(num_steps):
            start_t = i * step
            end_t   = min(start_t + dash_len, total_len)
            x1 = int(p1[0] + ux * start_t)
            y1 = int(p1[1] + uy * start_t)
            x2 = int(p1[0] + ux * end_t)
            y2 = int(p1[1] + uy * end_t)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
