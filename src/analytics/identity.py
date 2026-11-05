"""
Per-track identity smoothing & temporal feature bank fusion.

Face matches are noisy frame-to-frame (pose, blur, occlusion). Instead of
re-labelling a person every frame, each (camera, track):
1. Accumulates similarity-weighted votes with exponential decay.
2. Maintains an in-memory Feature Bank of the highest quality face embeddings (512-d).
3. Evaluates temporal similarity fusion against the enrolled face gallery to prevent
   false rejections or momentary identity dropouts during head turns or lighting shifts.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

UNKNOWN = "Unknown"


class TrackIdentityResolver:
    """
    Multi-frame identity resolver with exponential voting decay and
    in-memory temporal face embedding feature bank.
    """

    def __init__(
        self,
        decay: float = 0.85,
        min_votes: float = 1.0,
        switch_margin: float = 1.4,
        ttl_s: float = 10.0,
        hold_sec: float = 3.0,
        max_bank_size: int = 10,
    ):
        self.decay = decay
        self.min_votes = min_votes
        self.switch_margin = switch_margin
        self.ttl_s = ttl_s
        self.hold_sec = hold_sec
        self.max_bank_size = max_bank_size

        # (cam, track) -> {
        #   "votes": {name: score},
        #   "name": str,
        #   "sim": float,
        #   "seen": ts,
        #   "last_known_ts": ts,
        #   "feature_bank": List[Tuple[float, np.ndarray]],  # (quality/sim, 512-d unit vector)
        # }
        self._state: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def update(
        self,
        cam_id: str,
        track_id: int,
        name: str,
        similarity: float,
        now: Optional[float] = None,
        embedding: Optional[np.ndarray] = None,
        face_db: Optional[Any] = None,
    ) -> Tuple[str, float]:
        """
        Updates track identity with single-frame match and temporal feature bank fusion.
        """
        now = now or time.time()
        st = self._state.setdefault((cam_id, track_id), {
            "votes": defaultdict(float),
            "name": UNKNOWN,
            "sim": 0.0,
            "seen": now,
            "last_known_ts": 0.0,
            "feature_bank": [],
        })

        # ── 1. Update Feature Bank in RAM ────────────────────────────────────
        if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size > 0:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                unit_emb = (embedding / norm).astype(np.float32)
                quality_score = max(0.1, float(similarity))
                bank: List[Tuple[float, np.ndarray]] = st["feature_bank"]
                bank.append((quality_score, unit_emb))
                # Keep top N highest quality embeddings in memory
                if len(bank) > self.max_bank_size:
                    bank.sort(key=lambda item: item[0], reverse=True)
                    st["feature_bank"] = bank[: self.max_bank_size]

        # ── 2. Temporal Feature Fusion across gallery ────────────────────────
        # If single-frame result is uncertain or Unknown, check temporal feature bank
        if (name == UNKNOWN or similarity < 0.55) and face_db is not None and st["feature_bank"]:
            try:
                gallery = getattr(face_db, "_matrix", None)
                names = getattr(face_db, "_template_names", None)
                if gallery is not None and names and len(gallery) > 0:
                    # Multi-sample cosine similarity: (B, D) @ (D, G) -> (B, G)
                    bank_mat = np.stack([vec for _, vec in st["feature_bank"]])
                    sim_mat = np.dot(bank_mat, gallery.T)
                    # Max similarity per gallery template across all banked frames
                    max_sims = np.max(sim_mat, axis=0)
                    best_idx = int(np.argmax(max_sims))
                    best_sim = float(max_sims[best_idx])
                    best_name = names[best_idx]
                    match_thresh = getattr(face_db, "match_threshold", 0.52)
                    if best_sim >= match_thresh:
                        name = best_name
                        similarity = max(similarity, best_sim)
            except Exception:
                pass

        # ── 3. Vote Decay & Accumulation ─────────────────────────────────────
        votes = st["votes"]
        for k in list(votes):
            votes[k] *= self.decay

        if name != UNKNOWN:
            votes[name] += max(0.1, float(similarity))
            st["last_known_ts"] = now
        else:
            # Low vote for Unknown so momentary blurs/turns don't immediately wipe out a recognized person
            votes[name] += 0.15

        st["seen"] = now

        best = max(votes.items(), key=lambda kv: kv[1])
        current = st["name"]

        # Only promote to or switch identity if vote threshold is met
        if best[0] != current:
            if best[0] != UNKNOWN and best[1] >= self.min_votes:
                if current == UNKNOWN or best[1] >= self.switch_margin * votes.get(current, 0.0):
                    st["name"] = best[0]
            elif best[0] == UNKNOWN:
                # Revert to Unknown only if no positive match has occurred within hold_sec
                if now - st.get("last_known_ts", 0.0) > self.hold_sec:
                    st["name"] = UNKNOWN

        if st["name"] == name:
            st["sim"] = float(similarity)
        elif st["name"] != UNKNOWN and (now - st.get("last_known_ts", 0.0) <= self.hold_sec):
            # Smoothly decay the reported similarity during hold period
            st["sim"] = max(0.0, st["sim"] * 0.95)

        return st["name"], st["sim"]

    def get(self, cam_id: str, track_id: int) -> Optional[Tuple[str, float]]:
        st = self._state.get((cam_id, track_id))
        return (st["name"], st["sim"]) if st else None

    def purge(self, cam_id: str, active_tracks: Iterable[int], now: Optional[float] = None) -> None:
        now = now or time.time()
        active = set(active_tracks)
        for key in [k for k in self._state if k[0] == cam_id]:
            if key[1] not in active and now - self._state[key]["seen"] > self.ttl_s:
                del self._state[key]

    def clear(self, cam_id: Optional[str] = None) -> None:
        if cam_id is None:
            self._state.clear()
        else:
            for key in [k for k in self._state if k[0] == cam_id]:
                del self._state[key]
