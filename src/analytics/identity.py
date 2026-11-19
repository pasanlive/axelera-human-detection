"""
Per-track identity smoothing & temporal feature bank fusion with EMA and Hysteresis.

Implements:
1. Track-level Exponential Moving Average (EMA, alpha: 0.3) of L2-normalized embeddings.
2. Dual-threshold hysteresis state transitions:
   - match_high_thresh (0.55): required to lock/assign a new identity.
   - match_low_thresh (0.45): required to maintain an already-locked identity.
3. In-memory temporal feature bank of highest quality embeddings.
4. Auto-purge of stale tracks on track expiration.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

UNKNOWN = "Unknown"


class TrackIdentityResolver:
    """
    Multi-frame identity resolver with exponential voting decay, EMA embedding
    vector smoothing, and dual-threshold hysteresis.
    """

    def __init__(
        self,
        decay: float = 0.85,
        min_votes: float = 1.0,
        switch_margin: float = 1.4,
        ttl_s: float = 10.0,
        hold_sec: float = 3.0,
        max_bank_size: int = 10,
        ema_alpha: float = 0.3,
        match_high_thresh: float = 0.55,
        match_low_thresh: float = 0.45,
    ):
        self.decay = decay
        self.min_votes = min_votes
        self.switch_margin = switch_margin
        self.ttl_s = ttl_s
        self.hold_sec = hold_sec
        self.max_bank_size = max_bank_size
        self.ema_alpha = float(ema_alpha)
        self.match_high_thresh = float(match_high_thresh)
        self.match_low_thresh = float(match_low_thresh)

        # (cam, track) -> state dict
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
        Updates track identity with EMA embedding smoothing, temporal feature bank
        fusion, and dual-threshold hysteresis.
        """
        now = now or time.time()
        st = self._state.setdefault((cam_id, track_id), {
            "votes": defaultdict(float),
            "name": UNKNOWN,
            "sim": 0.0,
            "seen": now,
            "last_known_ts": 0.0,
            "feature_bank": [],
            "smoothed_embedding": None,
        })

        # ── 1. Update EMA Embedding & Feature Bank in RAM ────────────────────
        if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size > 0:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                unit_emb = (embedding / norm).astype(np.float32)

                # EMA embedding smoothing: (1 - alpha) * prev + alpha * curr
                prev_ema = st["smoothed_embedding"]
                if prev_ema is None:
                    st["smoothed_embedding"] = unit_emb.copy()
                else:
                    ema_vec = (1.0 - self.ema_alpha) * prev_ema + self.ema_alpha * unit_emb
                    ema_norm = np.linalg.norm(ema_vec)
                    if ema_norm > 0:
                        st["smoothed_embedding"] = (ema_vec / ema_norm).astype(np.float32)

                # Maintain feature bank of top quality embeddings
                quality_score = max(0.1, float(similarity))
                bank: List[Tuple[float, np.ndarray]] = st["feature_bank"]
                bank.append((quality_score, unit_emb))
                if len(bank) > self.max_bank_size:
                    bank.sort(key=lambda item: item[0], reverse=True)
                    st["feature_bank"] = bank[: self.max_bank_size]

        # ── 2. Temporal Feature Bank & EMA Gallery Matching ──────────────────
        if face_db is not None and (st["smoothed_embedding"] is not None or st["feature_bank"]):
            try:
                gallery = getattr(face_db, "_matrix", None)
                names = getattr(face_db, "_template_names", None)
                if gallery is not None and names and len(gallery) > 0:
                    # Cosine sim with EMA smoothed embedding
                    ema_sim = -1.0
                    ema_name = UNKNOWN
                    if st["smoothed_embedding"] is not None:
                        sims = np.dot(gallery, st["smoothed_embedding"])
                        best_idx = int(np.argmax(sims))
                        ema_sim = float(sims[best_idx])
                        ema_name = names[best_idx]

                    # Multi-sample cosine similarity across banked frames
                    bank_sim = -1.0
                    bank_name = UNKNOWN
                    if st["feature_bank"]:
                        bank_mat = np.stack([vec for _, vec in st["feature_bank"]])
                        sim_mat = np.dot(bank_mat, gallery.T)
                        max_sims = np.max(sim_mat, axis=0)
                        b_idx = int(np.argmax(max_sims))
                        bank_sim = float(max_sims[b_idx])
                        bank_name = names[b_idx]

                    best_sim = max(ema_sim, bank_sim)
                    best_cand = ema_name if ema_sim >= bank_sim else bank_name

                    # If fused match passes low threshold or rescues unknown, apply
                    if (name == UNKNOWN or similarity < self.match_high_thresh) and best_sim >= self.match_low_thresh:
                        name = best_cand
                        similarity = max(similarity, best_sim)
            except Exception:
                pass

        # ── 3. Vote Decay & Hysteresis State Updates ─────────────────────────
        votes = st["votes"]
        for k in list(votes):
            votes[k] *= self.decay

        current = st["name"]

        if current == UNKNOWN:
            # High threshold to establish a NEW locked identity
            if name != UNKNOWN and similarity >= self.match_high_thresh:
                votes[name] += max(0.1, float(similarity))
                if votes[name] >= self.min_votes:
                    st["name"] = name
                    st["sim"] = float(similarity)
                    st["last_known_ts"] = now
            else:
                votes[UNKNOWN] += 0.15
                st["sim"] = float(similarity)
        else:
            # Already locked to `current` identity
            if name == current:
                # Lower threshold is sufficient to maintain locked identity
                if similarity >= self.match_low_thresh:
                    votes[name] += max(0.1, float(similarity))
                    st["sim"] = float(similarity)
                    st["last_known_ts"] = now
                else:
                    # Below low threshold: check hold time before reverting
                    if now - st.get("last_known_ts", 0.0) <= self.hold_sec:
                        st["sim"] = max(0.0, st["sim"] * 0.95)
                    else:
                        votes[UNKNOWN] += 0.2
                        if votes.get(UNKNOWN, 0.0) > votes.get(current, 0.0):
                            st["name"] = UNKNOWN
                            st["sim"] = float(similarity)
            elif name != UNKNOWN:
                # Challenging candidate: require high threshold AND switch margin
                if similarity >= self.match_high_thresh:
                    votes[name] += max(0.1, float(similarity))
                    if votes[name] >= self.switch_margin * votes.get(current, 0.0) and votes[name] >= self.min_votes:
                        st["name"] = name
                        st["sim"] = float(similarity)
                        st["last_known_ts"] = now
                else:
                    votes[name] += 0.05
            else:
                # Unknown observation: rely on hold period
                votes[UNKNOWN] += 0.15
                if now - st.get("last_known_ts", 0.0) <= self.hold_sec:
                    st["sim"] = max(0.0, st["sim"] * 0.95)
                else:
                    st["name"] = UNKNOWN
                    st["sim"] = 0.0

        st["seen"] = now
        return st["name"], st["sim"]

    def get(self, cam_id: str, track_id: int) -> Optional[Tuple[str, float]]:
        st = self._state.get((cam_id, track_id))
        return (st["name"], st["sim"]) if st else None

    def purge(self, cam_id: str, active_tracks: Iterable[int], now: Optional[float] = None) -> None:
        """Purges stale track states when tracks expire."""
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
