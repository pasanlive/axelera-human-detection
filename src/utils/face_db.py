"""
FaceDatabase: Local Vector Database & Cosine Similarity Matcher for Facial Recognition.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

class FaceDatabase:
    """Stores enrolled face identity vectors and performs top-1 cosine matching."""

    def __init__(self, db_path: str = "data/face_db.json"):
        self.db_path = db_path
        self.identities: Dict[str, np.ndarray] = {}  # name -> array of 512-d embeddings
        self.load()

    def load(self):
        """Loads enrolled face gallery from disk."""
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.save()
            return

        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
                for name, vec_list in data.items():
                    self.identities[name] = np.array(vec_list, dtype=np.float32)
            print(f"[FACE DB] Loaded {len(self.identities)} enrolled identity face profiles.")
        except Exception as e:
            print(f"[FACE DB ERROR] Failed to load face database: {e}")

    def save(self):
        """Saves identity vectors to JSON file."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {name: vec.tolist() for name, vec in self.identities.items()}
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_identity(self, name: str, embedding: np.ndarray):
        """
        Enrolls a new face vector or updates an existing identity vector.
        :param name: Person's name / ID tag
        :param embedding: 512-d normalized face feature vector
        """
        # Normalize vector to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if name in self.identities:
            # Running average update
            self.identities[name] = 0.5 * (self.identities[name] + embedding)
            self.identities[name] /= np.linalg.norm(self.identities[name])
        else:
            self.identities[name] = embedding

        self.save()
        print(f"[FACE DB ENROLLED] Successfully enrolled identity: '{name}'.")

    def match(self, embedding: np.ndarray, threshold: float = 0.60) -> Tuple[str, float]:
        """
        Finds closest enrolled face match using cosine similarity.
        :param embedding: Query face embedding vector
        :param threshold: Minimum cosine similarity score required for positive match
        :return: Tuple of (Name, similarity_score) or ("Unknown", score)
        """
        if len(self.identities) == 0:
            return "Unknown", 0.0

        # Ensure unit norm
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        best_name = "Unknown"
        best_score = 0.0

        for name, ref_vec in self.identities.items():
            # Cosine similarity: dot product of normalized vectors
            sim = float(np.dot(embedding, ref_vec))
            if sim > best_score:
                best_score = sim
                best_name = name

        if best_score >= threshold:
            return best_name, best_score
        else:
            return "Unknown", best_score

    def list_identities(self) -> List[str]:
        return list(self.identities.keys())

    def remove_identity(self, name: str) -> bool:
        """Removes an enrolled identity from the database."""
        if name in self.identities:
            del self.identities[name]
            self.save()
            print(f"[FACE DB REMOVED] Removed identity: '{name}'.")
            return True
        return False
