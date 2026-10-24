"""
Per-track identity smoothing.

Face matches are noisy frame-to-frame (pose, blur, occlusion). Instead of
re-labelling a person every frame, each (camera, track) accumulates
similarity-weighted votes with exponential decay; the displayed identity only
changes once another name clearly dominates. Tracks that disappear are purged.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

UNKNOWN = "Unknown"


class TrackIdentityResolver:
    def __init__(self, decay: float = 0.85, min_votes: float = 1.2, switch_margin: float = 1.5,
                 ttl_s: float = 10.0):
        self.decay = decay
        self.min_votes = min_votes
        self.switch_margin = switch_margin
        self.ttl_s = ttl_s
        # (cam, track) -> {"votes": {name: score}, "name": str, "sim": float, "seen": ts}
        self._state: Dict[Tuple[str, int], Dict] = {}

    def update(self, cam_id: str, track_id: int, name: str, similarity: float,
               now: Optional[float] = None) -> Tuple[str, float]:
        now = now or time.time()
        st = self._state.setdefault((cam_id, track_id), {"votes": defaultdict(float), "name": UNKNOWN,
                                                         "sim": 0.0, "seen": now})
        votes = st["votes"]
        for k in list(votes):
            votes[k] *= self.decay
        votes[name] += max(0.05, float(similarity)) if name != UNKNOWN else 0.5
        st["seen"] = now

        best = max(votes.items(), key=lambda kv: kv[1])
        current = st["name"]
        if best[0] != current and best[1] >= self.min_votes:
            if current == UNKNOWN or best[1] >= self.switch_margin * votes.get(current, 0.0):
                st["name"] = best[0]
        if st["name"] == name:
            st["sim"] = float(similarity)
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
