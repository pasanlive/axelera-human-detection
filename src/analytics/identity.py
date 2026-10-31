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
    def __init__(self, decay: float = 0.85, min_votes: float = 1.0, switch_margin: float = 1.4,
                 ttl_s: float = 10.0, hold_sec: float = 3.0):
        self.decay = decay
        self.min_votes = min_votes
        self.switch_margin = switch_margin
        self.ttl_s = ttl_s
        self.hold_sec = hold_sec
        # (cam, track) -> {"votes": {name: score}, "name": str, "sim": float, "seen": ts, "last_known_ts": ts}
        self._state: Dict[Tuple[str, int], Dict] = {}

    def update(self, cam_id: str, track_id: int, name: str, similarity: float,
               now: Optional[float] = None) -> Tuple[str, float]:
        now = now or time.time()
        st = self._state.setdefault((cam_id, track_id), {
            "votes": defaultdict(float),
            "name": UNKNOWN,
            "sim": 0.0,
            "seen": now,
            "last_known_ts": 0.0,
        })
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
