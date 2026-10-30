"""
Dynamic capacity planner.

Chooses ONE quality tier for all cameras (resolution stream, pose model size,
SAHI tiling, frame-rate cap) so that the aggregate load fits three budgets:

* **AIPU**   - sum over cameras of ``fps * seconds-of-AIPU-per-frame`` must stay
               under ``target_utilization``. Per-model throughput comes from the
               Axelera Model Zoo reference FPS, derated for multi-model/multi-stream
               overhead (both configurable, calibrate with tools/benchmark_capacity.py).
* **VPU**    - RK3588 hardware decode throughput (Mpx/s). Decoding happens at the
               camera's native frame rate regardless of the inference FPS cap, so the
               budget decides main-vs-sub stream selection, not FPS.
* **Host**   - results/s the Python consumer (tracking glue, analytics, preview)
               can comfortably handle.

Few cameras  -> main stream, YOLOv8s-pose, tiling, 25 FPS.
Many cameras -> sub stream, YOLOv8n-pose, lower FPS.

Re-planning happens whenever cameras are added/removed or face/LPR are toggled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.voyager.network_builder import NetworkSpec

TILE_SIZE = 640


@dataclass(frozen=True)
class Tier:
    name: str
    stream: str  # "main" | "sub"
    pose_model: str  # "yolov8spose" | "yolov8npose"
    tiling: bool
    max_fps: int
    min_fps: int
    label: str = ""


TIERS: Tuple[Tier, ...] = (
    Tier("tracker", "sub", "yolov5m-v7-coco-tracker", False, 25, 10, "Sub stream · YOLOv5m + OC-SORT Tracker (Voyager SDK)"),
    Tier("ultra", "main", "yolov8spose", True, 25, 15, "Main stream · YOLOv8s-pose · SAHI tiling"),
    Tier("high", "main", "yolov8spose", False, 25, 12, "Main stream · YOLOv8s-pose"),
    Tier("standard", "main", "yolov8npose", False, 20, 10, "Main stream · YOLOv8n-pose"),
    Tier("balanced", "sub", "yolov8npose", False, 15, 8, "Sub stream · YOLOv8n-pose"),
    Tier("economy", "sub", "yolov8npose", False, 10, 5, "Sub stream · YOLOv8n-pose · reduced FPS"),
    Tier("minimal", "sub", "yolov8npose", False, 6, 3, "Sub stream · YOLOv8n-pose · minimum FPS"),
)
TIER_BY_NAME: Dict[str, Tier] = {t.name: t for t in TIERS}
PROFILES: Tuple[str, ...] = ("auto",) + tuple(t.name for t in TIERS)

# Axelera Model Zoo reference throughput (FPS, single model, 4 cores, Voyager 1.6).
DEFAULT_MODEL_FPS: Dict[str, float] = {
    "yolov5m-v7-coco-tracker": 350.0,
    "yolov8npose": 723.0,
    "yolov8spose": 471.0,
    "retinaface": 774.0,
    "facenet": 720.0,
    "plate_detector": 764.0,
    "lprnet": 1500.0,
}


@dataclass
class CameraSpec:
    id: str
    name: str
    source: str
    source_sub: Optional[str] = None
    main_resolution: Tuple[int, int] = (1920, 1080)
    sub_resolution: Tuple[int, int] = (640, 360)
    native_fps: float = 25.0
    features: FrozenSet[str] = frozenset()

    def stream_for(self, tier: Tier) -> Tuple[str, Tuple[int, int], str]:
        """(url, resolution, actual_stream) for the tier - falls back to main if no sub stream."""
        if tier.stream == "sub" and self.source_sub:
            return self.source_sub, tuple(self.sub_resolution), "sub"
        return self.source, tuple(self.main_resolution), "main"


@dataclass
class CapacityBudgets:
    model_fps: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MODEL_FPS))
    derating: float = 0.6
    """Fraction of zoo FPS achievable in a multi-model, multi-stream pipeline."""
    target_utilization: float = 0.85
    decode_mpx_per_s: float = 1000.0
    """RK3588 VPU: ~8K@30 H.264/H.265 ≈ 1000 Mpx/s."""
    host_results_per_s: float = 400.0
    expected_faces: float = 2.0
    expected_plates: float = 1.0
    # Off by default: on SDK 1.6.0 a tiled cascade (pose -> face) aborts natively with
    # "inplace_addstreamid: meta_key already exists" on the first frame. Opt-in only.
    enable_tiling: bool = False
    tile_overlap_pct: int = 10
    max_fps: int = 25

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CapacityBudgets":
        b = cls()
        for k in ("derating", "target_utilization", "decode_mpx_per_s", "host_results_per_s",
                  "expected_faces", "expected_plates"):
            if k in cfg:
                setattr(b, k, float(cfg[k]))
        for k in ("enable_tiling",):
            if k in cfg:
                setattr(b, k, bool(cfg[k]))
        for k in ("tile_overlap_pct", "max_fps"):
            if k in cfg:
                setattr(b, k, int(cfg[k]))
        b.model_fps.update({k: float(v) for k, v in (cfg.get("model_fps") or {}).items()})
        return b

    def eff_fps(self, model: str) -> float:
        return max(1e-6, self.model_fps.get(model, 500.0) * self.derating)


def tile_count(resolution: Sequence[int], overlap_pct: int, size: int = TILE_SIZE) -> int:
    """Number of AIPU passes for a tiled frame (tiles + one full-frame pass).

    Mirrors Ax::determine_overlap in the Voyager C++ tiler (overlap is a percentage).
    """
    def slices(n: int) -> int:
        overlapped = size - size * overlap_pct // 100
        return max(1, (n + overlapped - 1) // overlapped)

    w, h = int(resolution[0]), int(resolution[1])
    if w <= size and h <= size:
        return 1
    return slices(w) * slices(h) + 1


@dataclass
class PipelineGroup:
    spec: NetworkSpec
    camera_ids: List[str]
    sources: List[str]

    @property
    def network(self) -> str:
        return self.spec.name


@dataclass
class CapacityPlan:
    tier: Tier
    fps: int
    groups: List[PipelineGroup]
    aipu_utilization: float
    decode_mpx_per_s: float
    decode_utilization: float
    host_results_per_s: float
    overloaded: bool
    profile: str
    reasons: List[str] = field(default_factory=list)
    camera_streams: Dict[str, str] = field(default_factory=dict)
    """cam_id -> "main" | "sub" actually used."""

    @property
    def signature(self) -> Tuple:
        """Identity used to decide whether a running stream must be rebuilt."""
        return (self.tier.name, self.fps, tuple((g.network, tuple(g.camera_ids)) for g in self.groups))

    def structural_key(self) -> Tuple:
        """Tier/FPS identity (ignores camera membership)."""
        return (self.tier.name, self.fps)

    def group_for(self, cam_id: str) -> Optional[PipelineGroup]:
        for g in self.groups:
            if cam_id in g.camera_ids:
                return g
        return None

    def networks(self) -> List[str]:
        return [g.network for g in self.groups]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "tier": self.tier.name,
            "tier_label": self.tier.label,
            "fps": self.fps,
            "pose_model": self.tier.pose_model,
            "stream": self.tier.stream,
            "tiling": self.tier.tiling,
            "aipu_utilization": round(self.aipu_utilization, 3),
            "decode_mpx_per_s": round(self.decode_mpx_per_s, 1),
            "decode_utilization": round(self.decode_utilization, 3),
            "host_results_per_s": round(self.host_results_per_s, 1),
            "overloaded": self.overloaded,
            "reasons": list(self.reasons),
            "camera_streams": dict(self.camera_streams),
            "groups": [{"network": g.network, "cameras": list(g.camera_ids)} for g in self.groups],
        }


class CapacityPlanner:
    def __init__(self, budgets: Optional[CapacityBudgets] = None):
        self.budgets = budgets or CapacityBudgets()

    # ── cost model ──────────────────────────────────────────────────────────
    def frame_cost(self, cam: CameraSpec, tier: Tier) -> float:
        """Seconds of whole-AIPU time per processed frame for this camera."""
        b = self.budgets
        _, res, _ = cam.stream_for(tier)
        passes = tile_count(res, b.tile_overlap_pct) if (tier.tiling and b.enable_tiling) else 1
        cost = passes / b.eff_fps(tier.pose_model)
        if "face" in cam.features:
            cost += 1.0 / b.eff_fps("retinaface") + b.expected_faces / b.eff_fps("facenet")
        if "lpr" in cam.features:
            cost += 1.0 / b.eff_fps("plate_detector") + b.expected_plates / b.eff_fps("lprnet")
        return cost

    def decode_load(self, cams: Sequence[CameraSpec], tier: Tier) -> float:
        total = 0.0
        for c in cams:
            _, (w, h), _ = c.stream_for(tier)
            total += w * h * max(1.0, c.native_fps) / 1e6
        return total

    def _evaluate(self, cams: Sequence[CameraSpec], tier: Tier) -> Tuple[int, float, float, List[str]]:
        """Returns (fps, aipu_util_at_fps, decode_mpx, limiting reasons)."""
        b = self.budgets
        reasons: List[str] = []
        total_cost = sum(self.frame_cost(c, tier) for c in cams)
        fps_cap = min([tier.max_fps, b.max_fps] + [int(c.native_fps) for c in cams if c.native_fps > 0])
        aipu_bound = (b.target_utilization / total_cost) if total_cost > 0 else float("inf")
        host_bound = b.host_results_per_s / max(1, len(cams))
        fps = int(math.floor(min(fps_cap, aipu_bound, host_bound)))
        if aipu_bound < fps_cap:
            reasons.append(f"AIPU-bound at {aipu_bound:.1f} FPS/camera")
        if host_bound < fps_cap:
            reasons.append(f"host-bound at {host_bound:.1f} FPS/camera")
        decode = self.decode_load(cams, tier)
        return max(fps, 0), total_cost * max(fps, 0), decode, reasons

    # ── planning ────────────────────────────────────────────────────────────
    def plan(self, cams: Sequence[CameraSpec], profile: str = "auto") -> CapacityPlan:
        b = self.budgets
        cams = list(cams)
        candidates = [t for t in TIERS if (b.enable_tiling or not t.tiling) and t.name != "tracker"]
        if profile != "auto":
            if profile not in TIER_BY_NAME:
                raise ValueError(f"Unknown performance profile {profile!r}; expected one of {PROFILES}")
            candidates = [TIER_BY_NAME[profile]]

        chosen = None
        for tier in candidates:
            fps, util, decode, reasons = self._evaluate(cams, tier)
            decode_ok = decode <= b.decode_mpx_per_s
            if not cams or (fps >= tier.min_fps and decode_ok):
                chosen = (tier, fps, util, decode, reasons, False)
                break
        if chosen is None:
            # Nothing fits: run the last candidate at whatever it can sustain and flag it loudly.
            tier = candidates[-1]
            fps, util, decode, reasons = self._evaluate(cams, tier)
            fps = max(1, fps)
            util = sum(self.frame_cost(c, tier) for c in cams) * fps
            if decode > b.decode_mpx_per_s:
                reasons.append(f"VPU decode over budget ({decode:.0f}/{b.decode_mpx_per_s:.0f} Mpx/s) - "
                               "configure camera sub streams")
            if fps < tier.min_fps:
                reasons.append(f"below tier minimum of {tier.min_fps} FPS")
            chosen = (tier, fps, util, decode, reasons, True)

        tier, fps, util, decode, reasons, overloaded = chosen
        if not cams:
            fps = tier.max_fps

        groups: Dict[NetworkSpec, PipelineGroup] = {}
        streams: Dict[str, str] = {}
        for c in cams:
            spec = NetworkSpec(tier.pose_model, face="face" in c.features, lpr="lpr" in c.features)
            url, res, actual = c.stream_for(tier)
            streams[c.id] = actual
            if tier.tiling and b.enable_tiling and tile_count(res, b.tile_overlap_pct) > 1:
                url = f"tile[size={TILE_SIZE},overlap={b.tile_overlap_pct}]:{url}"
            g = groups.setdefault(spec, PipelineGroup(spec, [], []))
            g.camera_ids.append(c.id)
            g.sources.append(url)

        return CapacityPlan(
            tier=tier,
            fps=int(fps),
            groups=sorted(groups.values(), key=lambda g: g.network),
            aipu_utilization=util,
            decode_mpx_per_s=decode,
            decode_utilization=decode / b.decode_mpx_per_s if b.decode_mpx_per_s else 0.0,
            host_results_per_s=fps * len(cams),
            overloaded=overloaded,
            profile=profile,
            reasons=reasons,
            camera_streams=streams,
        )

    def required_specs(self, cams: Sequence[CameraSpec]) -> List[NetworkSpec]:
        """All networks any tier could need for these cameras (used by the deploy tool)."""
        specs = set()
        for tier in TIERS:
            for c in cams:
                specs.add(NetworkSpec(tier.pose_model, face="face" in c.features, lpr="lpr" in c.features))
        return sorted(specs, key=lambda s: s.name)
