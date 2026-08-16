"""
FaceRecognizer: Face Detection, Feature Embedding Extractor, and Identity Matcher.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.inference.voyager_engine import VoyagerEngine
from src.utils.face_db import FaceDatabase

class FaceRecognizer:
    """Extracts 512-d face embeddings and matches identities against FaceDatabase."""

    def __init__(self, config: Dict[str, Any], face_db: FaceDatabase):
        self.config = config
        self.face_db = face_db
        self.match_thresh = config.get("match_threshold", 0.60)
        self.input_size = tuple(config.get("input_size", [112, 112]))

        # Face feature embedder engine
        self.embedder_engine = VoyagerEngine(
            axm_path=config.get("embedder_axm"),
            onnx_path=config.get("embedder_onnx"),
            chip_id=config.get("chip_id", 0),
            num_cores=config.get("num_cores", 4)
        )

        # OpenCV Haar Cascade / DNN face detector as fast face localization fallback
        self.cascade_detector = None
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.cascade_detector = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.cascade_detector = None

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extracts normalized 512-d facial embedding vector from cropped face region.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        # Resize to 112x112 standard ArcFace input shape
        resized = cv2.resize(face_crop, self.input_size)
        # Normalize and transpose -> [1, 3, 112, 112]
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5  # ArcFace standard mean/std normalization [-1, 1]
        tensor = np.transpose(blob, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.embedder_engine.run(tensor)
        embedding = outputs[0].flatten()

        if self.embedder_engine.get_backend() == "virtual":
            # Generate deterministic synthetic embedding derived from image color statistics for testing
            np.random.seed(int(np.mean(face_crop)) if face_crop.size > 0 else 42)
            embedding = np.random.randn(512).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def recognize_faces_in_frame(self, frame: np.ndarray, person_bboxes: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Detects faces in frame (or within human bounding box crops) and identifies them.
        :return: List of dicts with 'bbox', 'name', 'similarity', 'embedding'
        """
        results = []
        h_frame, w_frame = frame.shape[:2]

        crops_to_process = []
        
        if person_bboxes:
            # Detect face within top portion (upper 35%) of each human bounding box crop
            for bbox in person_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                
                # Head region heuristic: top 35% of body box
                head_height = int((y2 - y1) * 0.35)
                if head_height > 20 and (x2 - x1) > 20:
                    head_crop = frame[y1:y1 + head_height, x1:x2]
                    crops_to_process.append(([x1, y1, x2, y1 + head_height], head_crop))
        elif self.cascade_detector is not None:
            # Global cascade face detection on full frame
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.cascade_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    crops_to_process.append(([x, y, x + w, y + h], frame[y:y + h, x:x + w]))
            except Exception:
                pass

        for bbox, face_crop in crops_to_process:
            if face_crop is None or face_crop.shape[0] < 15 or face_crop.shape[1] < 15:
                continue

            embedding = self.extract_embedding(face_crop)
            name, similarity = self.face_db.match(embedding, threshold=self.match_thresh)

            results.append({
                "bbox": bbox,
                "name": name,
                "similarity": similarity,
                "embedding": embedding
            })

        return results
