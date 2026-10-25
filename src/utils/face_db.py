"""
FaceDatabase: Local Vector Database & Cosine Similarity Matcher for Facial Recognition.

Embeddings are produced by the FaceNet (VGGFace2) model running on the Metis AIPU.
Embeddings from different models live in different vector spaces, so the file
records which model produced them; galleries from another model (e.g. the old
ArcFace/CPU pipeline) are backed up and must be re-enrolled.
"""

import json
import os
import shutil
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple

EMBEDDING_MODEL = "facenet-vggface2"
FORMAT_VERSION = 2


class FaceDatabase:
    """Stores enrolled face identity vectors and performs top-1 cosine matching."""

    def __init__(self, db_path: str = "data/face_db.json", model: str = EMBEDDING_MODEL, quiet: bool = False):
        self.db_path = db_path
        self.model = model
        self.quiet = quiet
        self.identities: Dict[str, np.ndarray] = {}  # name -> unit-norm embedding
        self._lock = threading.RLock()
        self._names: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # (K, D) stacked gallery for vectorised matching
        self.load(quiet=quiet)

    # ── persistence ──────────────────────────────────────────────────────────
    def load(self, quiet: Optional[bool] = None):
        """Loads enrolled face gallery from disk."""
        if quiet is None:
            quiet = self.quiet
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self.save()
            return

        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[FACE DB ERROR] Failed to load face database: {e}")
            return

        if isinstance(data, dict) and "identities" in data and "_meta" in data:
            file_model = data["_meta"].get("model")
            if file_model != self.model:
                self._backup_incompatible(f"model '{file_model}' != '{self.model}'")
                return
            entries = data["identities"]
        else:
            # Legacy (v1) flat {name: vector} file from the CPU/ArcFace pipeline.
            if data:
                self._backup_incompatible("legacy v1 format (embeddings from a different model)")
                return
            entries = {}

        with self._lock:
            for name, vec_list in entries.items():
                self.identities[name] = np.asarray(vec_list, dtype=np.float32)
            self._rebuild_index()
        if not quiet:
            print(f"[FACE DB] Loaded {len(self.identities)} enrolled identity face profiles ({self.model}).")

    def _backup_incompatible(self, reason: str):
        backup = os.path.splitext(self.db_path)[0] + ".legacy.json"
        shutil.copyfile(self.db_path, backup)
        print(f"[FACE DB WARNING] {reason}. Backed up to {backup}; starting with an empty gallery. "
              "Please re-enroll identities.")
        with self._lock:
            self.identities = {}
            self._rebuild_index()
        self.save()

    def save(self):
        """Saves identity vectors to JSON file."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._lock:
            data = {
                "_meta": {"version": FORMAT_VERSION, "model": self.model},
                "identities": {name: vec.tolist() for name, vec in self.identities.items()},
            }
        tmp = self.db_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.db_path)

    def _rebuild_index(self):
        self._names = list(self.identities.keys())
        self._matrix = np.stack([self.identities[n] for n in self._names]).astype(np.float32) \
            if self._names else None

    # ── mutation ─────────────────────────────────────────────────────────────
    def add_identity(self, name: str, embedding: np.ndarray):
        """
        Enrolls a new face vector or updates an existing identity vector.
        :param name: Person's name / ID tag
        :param embedding: 512-d normalized face feature vector
        """
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        # Normalize vector to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        with self._lock:
            if name in self.identities:
                # Running average update
                self.identities[name] = 0.5 * (self.identities[name] + embedding)
                self.identities[name] /= np.linalg.norm(self.identities[name])
            else:
                self.identities[name] = embedding
            self._rebuild_index()

        self.save()
        print(f"[FACE DB ENROLLED] Successfully enrolled identity: '{name}'.")

    def remove_identity(self, name: str) -> bool:
        """Removes an enrolled identity from the database."""
        with self._lock:
            if name not in self.identities:
                return False
            del self.identities[name]
            self._rebuild_index()
        self.save()
        print(f"[FACE DB REMOVED] Removed identity: '{name}'.")
        return True

    # ── matching ─────────────────────────────────────────────────────────────
    def match(self, embedding: np.ndarray, threshold: float = 0.60) -> Tuple[str, float]:
        """
        Finds closest enrolled face match using cosine similarity.
        :param embedding: Query face embedding vector
        :param threshold: Minimum cosine similarity score required for positive match
        :return: Tuple of (Name, similarity_score) or ("Unknown", score)
        """
        return self.match_batch(np.asarray(embedding, dtype=np.float32).reshape(1, -1), threshold)[0]

    def match_batch(self, embeddings: np.ndarray, threshold: float = 0.60) -> List[Tuple[str, float]]:
        """Vectorised top-1 cosine match for (N, D) query embeddings."""
        q = np.asarray(embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[0] == 0:
            return []
        with self._lock:
            names, gallery = self._names, self._matrix
        if gallery is None or gallery.shape[1] != q.shape[1]:
            return [("Unknown", 0.0)] * q.shape[0]
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        sims = q @ gallery.T  # (N, K)
        best = np.argmax(sims, axis=1)
        scores = sims[np.arange(q.shape[0]), best]
        return [(names[b] if s >= threshold else "Unknown", float(max(s, 0.0))) for b, s in zip(best, scores)]

    def list_identities(self) -> List[str]:
        with self._lock:
            return list(self.identities.keys())
