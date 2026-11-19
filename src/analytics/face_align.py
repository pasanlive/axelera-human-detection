"""
Face Quality Gating & 5-Point Landmark Alignment.

Implements:
1. 5-point facial landmark alignment using an affine similarity transform
   (cv2.estimateAffinePartial2D) against canonical ArcFace 112x112 reference coordinates.
2. Face quality gate:
   - Minimum face dimensions (min_face_size: 64 px).
   - Blur score calculation via Laplacian variance (min_blur_score: 50.0).
3. Clean interface & hook for model swap (e.g. SCRFD or OpenCV YuNet).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import cv2
import numpy as np

# Canonical ArcFace 112x112 5-point reference facial landmarks
# Coordinates correspond to [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
ARCFACE_REF_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],  # Left Eye
        [73.5318, 51.5014],  # Right Eye
        [56.0252, 71.7366],  # Nose Tip
        [41.5493, 92.3655],  # Left Mouth Corner
        [70.7299, 92.2041],  # Right Mouth Corner
    ],
    dtype=np.float32,
)


def compute_blur_score(crop: np.ndarray) -> float:
    """
    Computes blur score using variance of the Laplacian.
    Higher values indicate sharp edges; lower values (< 50.0) indicate motion or optical blur.
    """
    if crop is None or crop.size == 0:
        return 0.0

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def check_face_quality(
    crop: np.ndarray,
    min_face_size: int = 64,
    min_blur_score: float = 50.0,
) -> Tuple[bool, Optional[str], float]:
    """
    Evaluates whether a cropped face passes the quality gate.
    
    Checks:
    1. Dimensions: both width and height must be >= min_face_size.
    2. Sharpness: Laplacian variance blur score must be >= min_blur_score.

    Returns:
        (passes_quality, failure_reason, blur_score)
    """
    if crop is None or crop.size == 0:
        return False, "empty_crop", 0.0

    h, w = crop.shape[:2]
    if w < min_face_size or h < min_face_size:
        return False, f"size_too_small ({w}x{h} < {min_face_size}x{min_face_size})", 0.0

    blur_score = compute_blur_score(crop)
    if blur_score < min_blur_score:
        return False, f"blurry ({blur_score:.1f} < {min_blur_score:.1f})", blur_score

    return True, None, blur_score


def align_face_5point(
    image: np.ndarray,
    landmarks: Union[np.ndarray, list],
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Warps a face image to standard aligned coordinates using a 2D affine similarity transform.

    Args:
        image: Full frame or rough face bounding box BGR image.
        landmarks: 5 keypoints (x, y) or (x, y, conf) in the image's coordinate space.
                   Order: [left_eye, right_eye, nose, left_mouth, right_mouth].
        output_size: Target aligned crop size (width, height), default (112, 112).

    Returns:
        Aligned and cropped face image of shape (height, width, 3).
        Falls back to cv2.resize if alignment fails or landmarks are invalid.
    """
    if image is None or image.size == 0:
        return image

    src_pts = np.asarray(landmarks, dtype=np.float32)
    if src_pts.ndim == 2 and src_pts.shape[1] >= 2:
        src_pts = src_pts[:5, :2]
    else:
        # Invalid landmarks format - fallback to resize
        return cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)

    if len(src_pts) != 5:
        return cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)

    # Scale reference landmarks if target size differs from 112x112
    ref_pts = ARCFACE_REF_LANDMARKS_112.copy()
    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        ref_pts[:, 0] *= scale_x
        ref_pts[:, 1] *= scale_y

    try:
        # Estimate partial 2D affine transform (similarity transform: translation, rotation, uniform scale)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
        if M is not None:
            aligned = cv2.warpAffine(
                image,
                M,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            return aligned
    except Exception:
        pass

    # Fallback if transform could not be calculated
    return cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)


# ── Detector Swap / Extension Hook ───────────────────────────────────────────

class LandmarkDetectorHook:
    """
    Extensible interface for 5-point facial landmark detectors (e.g. YuNet, SCRFD).
    
    Architecture Note for Axelera Metis / OpenCV:
    - Current default in network_builder is MobileNetV2-based RetinaFace.
    - To enable real-time 5-point landmark output on the host CPU or Axelera AIPU,
      swap to SCRFD-500M / SCRFD-2.5G or OpenCV's built-in YuNet (cv2.FaceDetectorYN).
      YuNet outputs 15 values per face: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, conf].
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._detector = None
        if model_path and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                )
            except Exception:
                self._detector = None

    def detect_landmarks(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 5 landmarks from a face crop if detector is active.
        Returns: np.ndarray of shape (5, 2) or None.
        """
        if self._detector is None or face_bgr is None or face_bgr.size == 0:
            return None

        h, w = face_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(face_bgr)
        if faces is not None and len(faces) > 0:
            # Pick highest confidence face
            best_face = faces[0]
            # YuNet 5 points: right_eye, left_eye, nose, right_mouth, left_mouth
            # Note: ArcFace expects [left_eye, right_eye, nose, left_mouth, right_mouth]
            re = best_face[4:6]
            le = best_face[6:8]
            nt = best_face[8:10]
            rm = best_face[10:12]
            lm = best_face[12:14]
            landmarks = np.array([le, re, nt, lm, rm], dtype=np.float32)
            return landmarks
        return None
