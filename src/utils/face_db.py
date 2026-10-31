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
from typing import Any, Dict, List, Optional, Tuple

EMBEDDING_MODEL = "facenet-vggface2"
FORMAT_VERSION = 3
MAX_TEMPLATES_PER_IDENTITY = 10


class FaceDatabase:
    """Stores enrolled face identity vectors and performs top-1 multi-template cosine matching."""

    def __init__(self, db_path: str = "data/face_db.json", model: str = EMBEDDING_MODEL,
                 quiet: bool = False, max_templates: int = MAX_TEMPLATES_PER_IDENTITY):
        self.db_path = db_path
        self.model = model
        self.quiet = quiet
        self.max_templates = max_templates
        self.identities: Dict[str, List[np.ndarray]] = {}  # name -> list of unit-norm templates
        self.thumbnails: Dict[str, str] = {}  # name -> base64/data URL thumbnail
        self._lock = threading.RLock()
        self._template_names: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # (Total_Templates, D) stacked gallery
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
            if os.path.exists(self.db_path) and os.path.getsize(self.db_path) == 0:
                self.save()
                return
            with open(self.db_path, "r") as f:
                content = f.read().strip()
                if not content:
                    self.save()
                    return
                data = json.loads(content)
        except Exception as e:
            print(f"[FACE DB WARNING] Could not parse face database ({e}); reinitializing fresh gallery.")
            try:
                shutil.copyfile(self.db_path, f"{self.db_path}.corrupt.bak")
            except Exception:
                pass
            self.save()
            return

        if isinstance(data, dict) and "identities" in data and "_meta" in data:
            file_model = data["_meta"].get("model")
            if file_model != self.model:
                self._backup_incompatible(f"model '{file_model}' != '{self.model}'")
                return
            entries = data["identities"]
            thumbs = data.get("thumbnails", {})
        else:
            # Legacy (v1) flat {name: vector} file from the CPU/ArcFace pipeline.
            if data:
                self._backup_incompatible("legacy v1 format (embeddings from a different model)")
                return
            entries = {}
            thumbs = {}

        with self._lock:
            self.identities = {}
            self.thumbnails = dict(thumbs) if isinstance(thumbs, dict) else {}
            for name, val in entries.items():
                if not isinstance(val, list) or len(val) == 0:
                    continue
                # Support both flat 1D vector (v2) and list-of-vectors (v3 multi-template)
                if isinstance(val[0], (int, float)):
                    v = np.asarray(val, dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        self.identities[name] = [v / norm]
                elif isinstance(val[0], list):
                    vecs = []
                    for item in val:
                        v = np.asarray(item, dtype=np.float32)
                        norm = np.linalg.norm(v)
                        if norm > 0:
                            vecs.append(v / norm)
                    if vecs:
                        self.identities[name] = vecs[:self.max_templates]

            self._rebuild_index()

        if not quiet:
            total_templates = sum(len(v) for v in self.identities.values())
            print(f"[FACE DB] Loaded {len(self.identities)} enrolled identity face profiles "
                  f"({total_templates} templates, {self.model}).")

    def _backup_incompatible(self, reason: str):
        backup = os.path.splitext(self.db_path)[0] + ".legacy.json"
        shutil.copyfile(self.db_path, backup)
        print(f"[FACE DB WARNING] {reason}. Backed up to {backup}; starting with an empty gallery. "
              "Please re-enroll identities.")
        with self._lock:
            self.identities = {}
            self.thumbnails = {}
            self._rebuild_index()
        self.save()

    def save(self):
        """Saves identity vectors and thumbnails to JSON file."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._lock:
            data = {
                "_meta": {"version": FORMAT_VERSION, "model": self.model},
                "identities": {
                    name: [v.tolist() for v in vecs]
                    for name, vecs in self.identities.items()
                },
                "thumbnails": self.thumbnails,
            }
        tmp = self.db_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.db_path)

    def _rebuild_index(self):
        template_names: List[str] = []
        template_vecs: List[np.ndarray] = []
        for name, vecs in self.identities.items():
            for v in vecs:
                template_names.append(name)
                template_vecs.append(v)
        self._template_names = template_names
        self._matrix = np.stack(template_vecs).astype(np.float32) if template_vecs else None

    # ── mutation ─────────────────────────────────────────────────────────────
    def add_identity(self, name: str, embedding: np.ndarray, thumbnail: Optional[str] = None):
        """
        Enrolls a new face template vector or adds/refines a template for an existing identity.
        Maintains multiple distinct templates per person for high-accuracy multi-angle recognition.

        :param name: Person's name / ID tag
        :param embedding: 512-d face feature vector
        :param thumbnail: Optional base64/data URI face thumbnail
        """
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        with self._lock:
            if name not in self.identities:
                self.identities[name] = [embedding]
            else:
                existing = self.identities[name]
                sims = [float(np.dot(embedding, t)) for t in existing]
                max_sim = max(sims) if sims else 0.0

                if max_sim >= 0.96:
                    # Almost identical pose/angle: refine existing template via running average
                    idx = int(np.argmax(sims))
                    refined = 0.5 * (existing[idx] + embedding)
                    self.identities[name][idx] = refined / np.linalg.norm(refined)
                elif len(existing) < self.max_templates:
                    # New distinct angle/lighting: append new template to gallery
                    existing.append(embedding)
                else:
                    # Gallery limit reached: replace oldest template
                    existing.pop(0)
                    existing.append(embedding)

            if thumbnail:
                self.thumbnails[name] = thumbnail

            self._rebuild_index()

        self.save()
        total_samples = len(self.identities.get(name, []))
        print(f"[FACE DB ENROLLED] Successfully enrolled identity: '{name}' ({total_samples} templates).")

    def remove_identity(self, name: str) -> bool:
        """Removes an enrolled identity and its templates/thumbnail from the database."""
        with self._lock:
            if name not in self.identities:
                return False
            del self.identities[name]
            self.thumbnails.pop(name, None)
            self._rebuild_index()
        self.save()
        print(f"[FACE DB REMOVED] Removed identity: '{name}'.")
        return True

    # ── matching ─────────────────────────────────────────────────────────────
    def match(self, embedding: np.ndarray, threshold: float = 0.52) -> Tuple[str, float]:
        """
        Finds closest enrolled face match using multi-template cosine similarity.
        :param embedding: Query face embedding vector
        :param threshold: Minimum cosine similarity score required for positive match
        :return: Tuple of (Name, similarity_score) or ("Unknown", score)
        """
        return self.match_batch(np.asarray(embedding, dtype=np.float32).reshape(1, -1), threshold)[0]

    def match_batch(self, embeddings: np.ndarray, threshold: float = 0.52) -> List[Tuple[str, float]]:
        """Vectorised top-1 cosine match across all gallery templates."""
        q = np.asarray(embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[0] == 0:
            return []
        with self._lock:
            names, gallery = self._template_names, self._matrix
        if gallery is None or gallery.shape[1] != q.shape[1]:
            return [("Unknown", 0.0)] * q.shape[0]

        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        sims = q @ gallery.T  # (N, Total_Templates)
        best = np.argmax(sims, axis=1)
        scores = sims[np.arange(q.shape[0]), best]
        return [(names[b] if s >= threshold else "Unknown", float(max(s, 0.0)))
                for b, s in zip(best, scores)]

    def list_identities(self) -> List[str]:
        """Returns list of enrolled identity names."""
        with self._lock:
            return list(self.identities.keys())

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Returns rich profile for a single identity."""
        with self._lock:
            if name not in self.identities:
                return None
            return {
                "name": name,
                "sample_count": len(self.identities[name]),
                "thumbnail": self.thumbnails.get(name),
            }

    def get_profiles(self) -> List[Dict[str, Any]]:
        """Returns list of rich enrolled profiles."""
        with self._lock:
            return [
                {
                    "name": n,
                    "sample_count": len(self.identities[n]),
                    "thumbnail": self.thumbnails.get(n),
                }
                for n in self.identities
            ]
