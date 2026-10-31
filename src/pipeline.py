"""
Pipeline: End-to-End Multi-Camera Detection, Pose, and Face Recognition System Orchestrator.

Architecture (Voyager SDK "Option A" - the SDK owns the video pipeline):

    cameras ──► CapacityPlanner ──► VoyagerRuntime (InferenceStream, 1 pipeline / feature group)
                                         │  decode (RK3588 MPP) → preprocess → INT8 on Metis AIPU
                                         │  → C++ decoders/NMS → OC-SORT tracker
                                         ▼
                              on_result(cam, FrameAnalysis)   (metadata only)
                                         │
             face DB matching · identity voting · LPR logging · zone crossing
                                         │
                              PreviewHub (render only what is being watched)

No model ever runs on the host CPU. Toggling face recognition / LPR changes the
network composition and triggers a re-plan; the pose/detector toggles only
control what is drawn (one YOLOv8-pose model provides boxes AND keypoints).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.analytics.identity import TrackIdentityResolver
from src.render.preview import GRID, PreviewHub
from src.utils.face_db import FaceDatabase
from src.utils.face_history_db import FaceHistoryDatabase
from src.utils.plate_db import PlateDatabase
from src.utils.visualization import Visualizer
from src.voyager.capacity_planner import (
    PROFILES,
    TIER_BY_NAME,
    CameraSpec,
    CapacityBudgets,
    CapacityPlan,
    CapacityPlanner,
)
from src.voyager.result_parser import FrameAnalysis

# Models whose state changes the compiled network composition (=> re-plan).
NETWORK_FEATURES = {"face_recognizer": "face", "license_plate_recognizer": "lpr"}
MODEL_IDS = ("human_detector", "pose_estimator", "face_recognizer", "license_plate_recognizer", "zone_crossing")

# Legacy dashboard/config profile names -> capacity tiers.
LEGACY_PROFILE_MAP = {
    "high_precision": "high",
    "balanced": "standard",
    "ultra_light": "balanced",
    "yolo11_ultra_light": "balanced",
    "extreme_light": "economy",
}


def _res(v: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    try:
        return int(v[0]), int(v[1])
    except Exception:
        return default


class MultiCameraPipeline:
    """Orchestrates multi-camera Voyager inference, analytics and preview rendering."""

    def __init__(self, config: Dict[str, Any], simulate: bool = False,
                 runtime_factory: Optional[Callable[..., Any]] = None,
                 config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path
        self.simulate = simulate

        print("==========================================================")
        print("  Initializing Axelera Metis Multi-Camera Pipeline System ")
        print("==========================================================")

        models_cfg = config.get("models", {}) or {}
        face_rec_enabled = models_cfg.get("face_recognizer", {}).get("enabled", False)
        self.enabled_models: Dict[str, bool] = {
            "human_detector":           models_cfg.get("human_detector", {}).get("enabled", True),
            "pose_estimator":           models_cfg.get("pose_estimator", {}).get("enabled", True),
            "face_recognizer":          face_rec_enabled,
            "license_plate_recognizer": models_cfg.get("license_plate_recognizer", {}).get("enabled", False),
            "zone_crossing":            config.get("zone_crossing", {}).get("enabled", False),
        }
        self.camera_model_overrides: Dict[str, Dict[str, bool]] = {}
        for cam in config.get("cameras", []) or []:
            ov = cam.get("features") or {}
            if ov:
                self.camera_model_overrides[cam["id"]] = {k: bool(v) for k, v in ov.items() if k in MODEL_IDS}

        self.match_threshold = float(models_cfg.get("face_recognizer", {}).get("match_threshold", 0.52))
        self.ambiguity_margin = float(models_cfg.get("face_recognizer", {}).get("ambiguity_margin", 0.03))
        v_thresh = config.get("voyager", {}).get("thresholds", {}) or {}
        self.face_conf = float(v_thresh.get("face_conf", 0.65))
        self.min_face_px = int(v_thresh.get("min_face_px", 48))
        self.require_person_body = bool(v_thresh.get("require_person_body", True))
        self.plate_log_interval_s = float(models_cfg.get("license_plate_recognizer", {}).get("log_interval_s", 10.0))

        # ── Databases ────────────────────────────────────────────────────────
        self.face_db = FaceDatabase(
            db_path=config.get("face_db", {}).get("path", "data/face_db.json"),
            quiet=not face_rec_enabled,
        )
        self.face_history_db = FaceHistoryDatabase(
            db_path=config.get("face_db", {}).get("history_db_path", "data/face_history.db")
        )
        self.plate_db = PlateDatabase(db_path=config.get("plate_db", {}).get("path", "data/plate_db.json"))

        # ── Analytics ────────────────────────────────────────────────────────
        self.identity = TrackIdentityResolver()
        self._zone_crossing: Optional[Any] = None
        if self.enabled_models["zone_crossing"]:
            self._ensure_zone_crossing()
        self._plate_last_logged: Dict[Tuple[str, str], float] = {}

        # ── Capacity planning ────────────────────────────────────────────────
        cap_cfg = config.get("capacity", {}) or {}
        self.performance_profile = self._normalise_profile(cap_cfg.get("profile", "auto"))
        self.planner = CapacityPlanner(CapacityBudgets.from_config(cap_cfg))
        self.plan: Optional[CapacityPlan] = None

        # ── Rendering ────────────────────────────────────────────────────────
        self.visualizer = Visualizer(config)
        pv = config.get("preview", {}) or {}
        self.preview = PreviewHub(
            renderer=lambda frame, packet: self.visualizer.draw_frame(frame=frame, **packet),
            preview_fps=pv.get("fps", 12), grid_fps=pv.get("grid_fps", 6),
            max_width=pv.get("max_width", 1280), jpeg_quality=pv.get("jpeg_quality", 75),
        )

        # ── Runtime state ────────────────────────────────────────────────────
        self._state_lock = threading.RLock()
        self._latest: Dict[str, Dict[str, Any]] = {}  # cam -> {persons, faces, plates, ts}
        self.events: List[Dict[str, Any]] = []
        self._replan_lock = threading.Lock()
        self._replan_evt = threading.Event()
        self._replan_thread: Optional[threading.Thread] = None
        self.is_running = False

        if runtime_factory is None:
            if simulate:
                from src.voyager.stream_runtime import SimulatedRuntime
                runtime_factory = lambda on_result, on_event: SimulatedRuntime(on_result, on_event)  # noqa: E731
            else:
                from src.voyager.stream_runtime import VoyagerRuntime
                vcfg = config.get("voyager", {}) or {}
                runtime_factory = lambda on_result, on_event: VoyagerRuntime(vcfg, on_result, on_event)  # noqa: E731
        self.runtime = runtime_factory(self._on_result, self._on_event)
        print(f"[PIPELINE] Inference backend: {self.runtime.backend}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_profile(profile: str) -> str:
        # Current tier names win: "balanced" is both a legacy alias and a tier.
        if profile in PROFILES:
            return profile
        profile = LEGACY_PROFILE_MAP.get(profile, profile)
        return profile if profile in PROFILES else "auto"

    def _ensure_zone_crossing(self):
        if self._zone_crossing is None:
            zones = self.config.get("zone_crossing", {}).get("zones", []) or []
            from src.analytics.zone_crossing import ZoneCrossingDetector
            self._zone_crossing = ZoneCrossingDetector(zones)

    def _camera_cfgs(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        cams = self.config.get("cameras", []) or []
        return [c for c in cams if c.get("enabled", True) or not enabled_only]

    def _camera_features(self, cam_id: str) -> frozenset:
        feats = set(feat for model, feat in NETWORK_FEATURES.items() if self.is_model_enabled(model, cam_id))
        if self.is_model_enabled("pose_estimator", cam_id):
            feats.add("pose")
        return frozenset(feats)

    def camera_specs(self) -> List[CameraSpec]:
        specs = []
        for c in self._camera_cfgs():
            specs.append(CameraSpec(
                id=c["id"],
                name=c.get("name", c["id"]),
                source=c["source"],
                source_sub=c.get("source_sub"),
                main_resolution=_res(c.get("main_resolution"), (1920, 1080)),
                sub_resolution=_res(c.get("sub_resolution"), (640, 360)),
                native_fps=float(c.get("native_fps", 25)),
                features=self._camera_features(c["id"]),
            ))
        return specs

    # ─────────────────────────────────────────────────────────────────────────
    # Model enable/disable
    # ─────────────────────────────────────────────────────────────────────────

    def is_model_enabled(self, model_name: str, cam_id: Any = None) -> bool:
        """Checks whether a specific model is enabled globally or for a specific camera."""
        if cam_id and cam_id in self.camera_model_overrides:
            cam_overrides = self.camera_model_overrides[cam_id]
            if model_name in cam_overrides:
                return cam_overrides[model_name]
        return self.enabled_models.get(model_name, False)

    def set_model_enabled(self, model_name: str, enabled: bool, cam_id: Any = None):
        """Enables/disables a model globally or per camera. Face/LPR changes re-plan the network."""
        if model_name not in MODEL_IDS:
            raise ValueError(f"Unknown model '{model_name}'. Valid: {MODEL_IDS}")
        enabled = bool(enabled)
        if cam_id and cam_id != "all":
            self.camera_model_overrides.setdefault(cam_id, {})[model_name] = enabled
            print(f"[PIPELINE ADMIN] Model '{model_name}' on stream '{cam_id}' set to: {'ENABLED' if enabled else 'DISABLED'}")
        else:
            self.enabled_models[model_name] = enabled
            for overrides in self.camera_model_overrides.values():
                overrides.pop(model_name, None)
            print(f"[PIPELINE ADMIN] Model '{model_name}' globally set to: {'ENABLED' if enabled else 'DISABLED'}")

        if model_name == "zone_crossing" and enabled:
            self._ensure_zone_crossing()
        if model_name == "face_recognizer" and not enabled:
            self.identity.clear(None if not cam_id or cam_id == "all" else cam_id)
        if model_name in NETWORK_FEATURES or model_name == "pose_estimator":
            self.request_replan()

    def get_face_config(self) -> Dict[str, Any]:
        """Returns active face detection and recognition precision thresholds."""
        return {
            "match_threshold": self.match_threshold,
            "ambiguity_margin": self.ambiguity_margin,
            "face_conf": self.face_conf,
            "min_face_px": self.min_face_px,
            "require_person_body": self.require_person_body,
        }

    def configure_face_recognizer(self, match_threshold: Optional[float] = None,
                                  face_conf: Optional[float] = None,
                                  min_face_px: Optional[int] = None,
                                  ambiguity_margin: Optional[float] = None,
                                  require_person_body: Optional[bool] = None) -> Dict[str, Any]:
        """Dynamically tunes face detector and recognition matching precision."""
        if match_threshold is not None:
            self.match_threshold = max(0.20, min(0.95, float(match_threshold)))
        if ambiguity_margin is not None:
            self.ambiguity_margin = max(0.0, min(0.30, float(ambiguity_margin)))
        if face_conf is not None:
            self.face_conf = max(0.10, min(0.99, float(face_conf)))
        if min_face_px is not None:
            self.min_face_px = max(24, min(300, int(min_face_px)))
        if require_person_body is not None:
            self.require_person_body = bool(require_person_body)
        print(f"[FACE TUNING] Updated config: match_thresh={self.match_threshold:.2f}, "
              f"ambiguity_margin={self.ambiguity_margin:.2f}, face_conf={self.face_conf:.2f}, "
              f"min_face_px={self.min_face_px}, require_body={self.require_person_body}")
        return self.get_face_config()

    # ─────────────────────────────────────────────────────────────────────────
    # Capacity planning
    # ─────────────────────────────────────────────────────────────────────────

    def compute_plan(self) -> CapacityPlan:
        return self.planner.plan(self.camera_specs(), self.performance_profile)

    def replan_now(self) -> CapacityPlan:
        """Re-plans synchronously and applies the plan to the runtime if running."""
        with self._replan_lock:
            plan = self.compute_plan()
            prev = self.plan
            self.plan = plan
            self.preview.set_camera_order([c.id for c in self.camera_specs()])
            if plan.overloaded:
                self._on_event(None, "warning", "Capacity overloaded: " + "; ".join(plan.reasons))
            if self.is_running and (prev is None or prev.signature != plan.signature):
                specs = {c.id: c for c in self.camera_specs()}
                self.runtime.apply_plan(plan, specs)
                print(f"[CAPACITY] tier={plan.tier.name} fps={plan.fps} cams={len(specs)} "
                      f"aipu={plan.aipu_utilization:.0%} decode={plan.decode_utilization:.0%} "
                      f"networks={plan.networks()}")
            return plan

    def request_replan(self) -> None:
        """Debounced asynchronous re-plan (stream rebuilds can take seconds)."""
        if not self.is_running:
            self.plan = self.compute_plan()
            return
        self._replan_evt.set()

    def _replan_worker(self):
        while self.is_running:
            if not self._replan_evt.wait(timeout=0.5):
                continue
            time.sleep(0.3)  # coalesce bursts of toggles
            self._replan_evt.clear()
            try:
                self.replan_now()
            except Exception as e:
                self._on_event(None, "error", f"Re-plan failed: {e}")

    def get_capacity_plan(self) -> Dict[str, Any]:
        plan = self.plan or self.compute_plan()
        d = plan.to_dict()
        d["profiles"] = list(PROFILES)
        d["camera_count"] = len(self._camera_cfgs())
        return d

    def set_performance_profile(self, profile: str) -> Dict[str, Any]:
        profile = self._normalise_profile(profile)
        self.performance_profile = profile
        self.config.setdefault("capacity", {})["profile"] = profile
        self._persist_config()
        self.request_replan()
        return self.get_capacity_plan()

    def reload_detector_model(self, model_profile: str, *args, detect_interval: Optional[int] = None, **kwargs):
        """Backward-compatible shim for the legacy detector-profile API."""
        self.set_performance_profile(model_profile)

    # ─────────────────────────────────────────────────────────────────────────
    # Camera management
    # ─────────────────────────────────────────────────────────────────────────

    def add_camera(self, cam_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not cam_cfg.get("source"):
            raise ValueError("Camera 'source' is required")
        cams = self.config.setdefault("cameras", [])
        cam_id = cam_cfg.get("id") or f"cam_{len(cams) + 1:02d}"
        if any(c["id"] == cam_id for c in cams):
            raise ValueError(f"Camera id '{cam_id}' already exists")
        entry = {"id": cam_id, "name": cam_cfg.get("name", cam_id), "enabled": cam_cfg.get("enabled", True),
                 "source": cam_cfg["source"]}
        for k in ("source_sub", "main_resolution", "sub_resolution", "native_fps", "features"):
            if cam_cfg.get(k) is not None:
                entry[k] = cam_cfg[k]
        cams.append(entry)
        self._persist_config()
        self.request_replan()
        return entry

    def remove_camera(self, cam_id: str) -> bool:
        cams = self.config.get("cameras", []) or []
        keep = [c for c in cams if c["id"] != cam_id]
        if len(keep) == len(cams):
            return False
        self.config["cameras"] = keep
        self.camera_model_overrides.pop(cam_id, None)
        self.identity.clear(cam_id)
        with self._state_lock:
            self._latest.pop(cam_id, None)
        self._persist_config()
        self.request_replan()
        return True

    def set_camera_enabled(self, cam_id: str, enabled: bool) -> bool:
        for c in self.config.get("cameras", []) or []:
            if c["id"] == cam_id:
                c["enabled"] = bool(enabled)
                self._persist_config()
                self.request_replan()
                return True
        return False

    def get_camera_status(self) -> List[Dict[str, Any]]:
        out = []
        plan = self.plan
        for c in self._camera_cfgs(enabled_only=False):
            cid = c["id"]
            group = plan.group_for(cid) if plan else None
            state = self.runtime.camera_state.get(cid, "stopped" if not c.get("enabled", True) else "pending")
            out.append({
                "id": cid,
                "name": c.get("name", cid),
                "enabled": c.get("enabled", True),
                "fps": round(float(self.runtime.camera_fps.get(cid, 0.0)), 1),
                "is_opened": state == "live",
                "state": state,
                "stream": plan.camera_streams.get(cid) if plan else None,
                "network": group.network if group else None,
                "features": sorted(self._camera_features(cid)),
            })
        return out

    def _persist_config(self):
        if not self.config_path or not os.path.exists(self.config_path):
            return
        try:
            import yaml
            with open(self.config_path, "r") as f:
                disk = yaml.safe_load(f) or {}
            disk["cameras"] = self.config.get("cameras", [])
            disk.setdefault("capacity", {})["profile"] = self.performance_profile
            with open(self.config_path, "w") as f:
                yaml.safe_dump(disk, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"[CONFIG SAVE NOTICE] Failed to persist config: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Result handling (runs on the runtime consumer thread - keep it light)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_event(self, cam_id: Optional[str], kind: str, message: str):
        ev = {"ts": time.time(), "camera_id": cam_id, "type": kind, "message": message}
        with self._state_lock:
            self.events.append(ev)
            self.events = self.events[-200:]
        print(f"[RUNTIME {kind.upper()}]{' ' + cam_id if cam_id else ''} {message}")

    def _on_result(self, cam_id: str, fa: FrameAnalysis, get_image: Callable[[], Optional[np.ndarray]]):
        now = fa.timestamp
        persons = fa.persons
        track_ids = [p["track_id"] for p in persons if p["track_id"] is not None]

        # Faces → identities (vectorised gallery match + per-track voting)
        faces_out: List[Dict[str, Any]] = []
        face_det_on = self.is_model_enabled("face_detector", cam_id) or self.is_model_enabled("face_recognizer", cam_id)
        face_rec_on = self.is_model_enabled("face_recognizer", cam_id)

        cached_frame: Optional[np.ndarray] = None

        def _get_frame():
            nonlocal cached_frame
            if cached_frame is None:
                cached_frame = get_image()
            return cached_frame

        def _crop_face_thumb(b):
            fr = _get_frame()
            if fr is None:
                return None
            try:
                import base64
                import cv2
                fh, fw = fr.shape[:2]
                bx = [b[0] * fw, b[1] * fh, b[2] * fw, b[3] * fh] if max(b) <= 1.05 and fw > 0 else b
                x1, y1, x2, y2 = map(int, bx)
                pw, ph = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
                cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
                cx2, cy2 = min(fw, x2 + pw), min(fh, y2 + ph)
                if cx2 > cx1 and cy2 > cy1:
                    crop = cv2.resize(fr[cy1:cy2, cx1:cx2], (112, 112))
                    _, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
            except Exception:
                pass
            return None

        if fa.faces and face_det_on:
            min_face_px = self.min_face_px
            face_conf_thresh = self.face_conf

            # Filter faces to eliminate false positives:
            # 1. Confidence threshold
            # 2. Aspect ratio (human-like bounding box, 0.48 <= w/h <= 1.42)
            # 3. Minimum pixel dimensions
            # 4. Body association: if require_person_body is True, face must belong to a detected person
            #    or have high detection confidence (>= 0.82) to avoid background texture false positives
            valid_faces = []
            for f in fa.faces:
                conf = float(f.get("confidence", 0.0))
                if conf < face_conf_thresh:
                    continue
                w_f = f["bbox"][2] - f["bbox"][0]
                h_f = f["bbox"][3] - f["bbox"][1]
                if w_f <= 0 or h_f <= 0:
                    continue
                aspect = float(w_f / h_f)
                if aspect < 0.48 or aspect > 1.42:
                    continue
                if self.require_person_body and f.get("person_index") is None and conf < 0.82:
                    continue
                valid_faces.append(f)
            fa.faces = valid_faces

            with_emb = [f for f in fa.faces if f.get("embedding") is not None
                        and (f["bbox"][2] - f["bbox"][0]) >= min_face_px
                        and (f["bbox"][3] - f["bbox"][1]) >= min_face_px] if face_rec_on else []
            matches = self.face_db.match_batch(
                np.stack([f["embedding"] for f in with_emb]),
                threshold=self.match_threshold,
                ambiguity_margin=self.ambiguity_margin
            ) if with_emb else []
            match_map = {id(f): (name, sim) for f, (name, sim) in zip(with_emb, matches)}
            cam_name = self._cam_name(cam_id)

            for f in fa.faces:
                w_f = f["bbox"][2] - f["bbox"][0]
                h_f = f["bbox"][3] - f["bbox"][1]
                is_large_enough = (w_f >= min_face_px and h_f >= min_face_px)

                name, sim = match_map.get(id(f), ("Unknown", 0.0))
                tid = None
                if f["person_index"] is not None and f["person_index"] < len(persons):
                    tid = persons[f["person_index"]]["track_id"]
                if tid is not None and face_rec_on and is_large_enough:
                    name, sim = self.identity.update(cam_id, tid, name, sim, now)
                    persons[f["person_index"]]["identity"] = name

                thumb = _crop_face_thumb(f["bbox"]) if (face_rec_on and is_large_enough) else None
                faces_out.append({"bbox": f["bbox"], "name": name, "similarity": sim,
                                  "track_id": tid, "confidence": f.get("confidence", 0.0),
                                  "embedding": f.get("embedding"), "thumbnail": thumb})

                # Face logging to SQLite DB & real-time notification with deduplication
                if face_rec_on and is_large_enough:
                    is_known = bool(name and name != "Unknown" and sim >= self.match_threshold)
                    is_new, det_count, sighting = self.face_history_db.log_sighting(
                        camera_id=cam_id,
                        name=name if is_known else "Unidentified",
                        is_identified=is_known,
                        similarity=sim,
                        confidence=f.get("confidence", 0.0),
                        track_id=tid,
                        bbox=f["bbox"],
                        thumbnail=thumb,
                        embedding=f.get("embedding"),
                        timestamp=now,
                    )

                    with self._state_lock:
                        target_ev = None
                        if is_known:
                            for ev in reversed(self.events):
                                if ev.get("type") == "face" and ev.get("camera_id") == cam_id and ev.get("name") == name:
                                    if now - ev.get("ts", 0) <= 60.0:
                                        target_ev = ev
                                        break
                        else:
                            fid = sighting.get("face_id")
                            for ev in reversed(self.events):
                                if ev.get("type") == "face_unidentified" and ev.get("camera_id") == cam_id:
                                    if (fid and ev.get("face_id") == fid) or (tid is not None and ev.get("track_id") == tid):
                                        if now - ev.get("ts", 0) <= 60.0:
                                            target_ev = ev
                                            break

                        if target_ev is not None:
                            target_ev["detection_count"] = det_count
                            target_ev["ts"] = now
                            if is_known:
                                target_ev["similarity"] = max(target_ev.get("similarity", 0.0), sim)
                                target_ev["message"] = f"Recognized '{name}' ({sim * 100:.0f}%) on {cam_name}"
                            else:
                                target_ev["message"] = f"Unidentified face on {cam_name}"
                            if thumb and not target_ev.get("thumbnail"):
                                target_ev["thumbnail"] = thumb
                        else:
                            import uuid
                            new_ev = {
                                "id": "ev_" + uuid.uuid4().hex[:8],
                                "ts": now,
                                "camera_id": cam_id,
                                "type": "face" if is_known else "face_unidentified",
                                "message": f"Recognized '{name}' ({sim * 100:.0f}%) on {cam_name}" if is_known else f"Unidentified face on {cam_name}",
                                "name": name if is_known else "Unidentified",
                                "similarity": sim,
                                "detection_count": det_count,
                                "thumbnail": thumb,
                                "unidentified": not is_known,
                                "face_id": sighting.get("face_id"),
                                "track_id": tid,
                            }
                            self.events.append(new_ev)
                            self.events = self.events[-200:]

        if face_rec_on:
            self.identity.purge(cam_id, track_ids, now)

        # Plates → throttled logging
        plates_out = fa.plates if self.is_model_enabled("license_plate_recognizer", cam_id) else []
        for p in plates_out:
            text = p.get("plate_text", "")
            if len(text) >= 2:
                key = (cam_id, text)
                if now - self._plate_last_logged.get(key, 0.0) >= self.plate_log_interval_s:
                    self._plate_last_logged[key] = now
                    self.plate_db.log_plate(text, cam_id, p.get("confidence", 0.0), 1.0)

        # Zone crossing on tracked centroids (zones are defined in main-stream pixels)
        zone_on = self.is_model_enabled("zone_crossing", cam_id) and self._zone_crossing is not None
        if zone_on and self._zone_crossing.has_zones_for_camera(cam_id):
            sx, sy = self._zone_scale(cam_id, fa.frame_size)
            for p in persons:
                if p["track_id"] is not None:
                    b = p["bbox"]
                    self._zone_crossing.update(cam_id, p["track_id"],
                                               ((b[0] + b[2]) / 2.0 * sx, (b[1] + b[3]) / 2.0 * sy))
            self._zone_crossing.purge_stale_tracks(cam_id, track_ids)

        with self._state_lock:
            self._latest[cam_id] = {"ts": now, "persons": len(persons), "faces": faces_out,
                                    "faces_detected": len(faces_out),
                                    "known_faces": sum(1 for f in faces_out if f["name"] != "Unknown"),
                                    "plates": len(plates_out), "frame_size": fa.frame_size}

        # Preview: only download/draw the frame when somebody is watching
        if self.preview.wants_frame(cam_id, now):
            frame = cached_frame if cached_frame is not None else get_image()
            if frame is not None:
                self.preview.submit(cam_id, frame, self._render_packet(cam_id, persons, faces_out, plates_out,
                                                                       zone_on, fa.frame_size))

    def _zone_scale(self, cam_id: str, frame_size: Tuple[int, int]) -> Tuple[float, float]:
        """Factor mapping decoded-frame pixels to main-stream pixels (zone coordinates)."""
        fw, fh = frame_size
        if not fw or not fh:
            return 1.0, 1.0
        cam_cfg = next((c for c in self._camera_cfgs(False) if c["id"] == cam_id), {})
        mw, mh = _res(cam_cfg.get("main_resolution"), (fw, fh))
        return mw / fw, mh / fh

    def _render_packet(self, cam_id, persons, faces, plates, zone_on, frame_size=(0, 0)) -> Dict[str, Any]:
        show_boxes = self.is_model_enabled("human_detector", cam_id)
        show_pose = self.is_model_enabled("pose_estimator", cam_id)
        cam_cfg = next((c for c in self._camera_cfgs(False) if c["id"] == cam_id), {})
        return {
            "detections": [{"bbox": p["bbox"], "confidence": p["confidence"], "track_id": p["track_id"]}
                           for p in persons] if show_boxes else [],
            "poses": [{"keypoints": p["keypoints"]} for p in persons
                      if p.get("keypoints") is not None] if show_pose else [],
            "faces": [{"bbox": f["bbox"], "name": f["name"], "similarity": f["similarity"]} for f in faces],
            "plates": plates,
            "zones": self._scaled_zones(cam_id, frame_size) if zone_on else [],
            "zone_counts": self._zone_crossing.get_zone_counts(cam_id) if zone_on else {},
            "stream_title": cam_cfg.get("name", cam_id),
            "fps": float(self.runtime.camera_fps.get(cam_id, 0.0)),
        }

    def _scaled_zones(self, cam_id: str, frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        zones = self._zone_crossing.get_zones_for_camera(cam_id)
        sx, sy = self._zone_scale(cam_id, frame_size)
        if sx == 1.0 and sy == 1.0:
            return zones
        return [dict(z, points=[[int(x / sx), int(y / sy)] for x, y in z.get("points", [])]) for z in zones]

    # ─────────────────────────────────────────────────────────────────────────
    # Enrollment
    # ─────────────────────────────────────────────────────────────────────────

    def enroll_identity(self, name: str, image: Optional[np.ndarray] = None,
                        cam_id: Optional[str] = None) -> Tuple[bool, str]:
        """Enrolls from an uploaded image (runs RetinaFace+FaceNet on the AIPU) or from
        the largest face currently visible on ``cam_id`` (no extra inference)."""
        if not name:
            return False, "Identity name is required"
        if image is None and not self.is_model_enabled("face_recognizer"):
            return False, "Face recognition must be enabled to enroll directly from the live camera stream."
        emb = None
        thumb = None
        if image is not None:
            try:
                emb = self.runtime.embed_face_image(image)
                import base64
                import cv2
                ih, iw = image.shape[:2]
                crop_sz = min(ih, iw)
                cy, cx = ih // 2, iw // 2
                crop = image[max(0, cy - crop_sz // 2):min(ih, cy + crop_sz // 2),
                             max(0, cx - crop_sz // 2):min(iw, cx + crop_sz // 2)]
                resized = cv2.resize(crop, (112, 112))
                _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                thumb = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
            except Exception as e:
                return False, f"Enrollment inference failed: {e}"
            if emb is None:
                return False, "No face detected in the image"
        else:
            with self._state_lock:
                cams = [cam_id] if cam_id and cam_id != "all" else list(self._latest)
                faces = [f for c in cams for f in self._latest.get(c, {}).get("faces", [])
                         if f.get("embedding") is not None]
            if not faces:
                return False, "No face with an embedding is currently visible (is face recognition enabled?)"
            best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            emb = best["embedding"]
            thumb = best.get("thumbnail")
        self.face_db.add_identity(name, emb, thumbnail=thumb)
        return True, f"Successfully enrolled '{name}'."

    def enroll_unidentified_face(self, face_id: str, name: str) -> Tuple[bool, str]:
        """Enrolls an unidentified face profile from SQLite history by its face_id."""
        if not name or not name.strip():
            return False, "Identity name is required"
        name = name.strip()
        emb = self.face_history_db.get_unidentified_embedding(face_id)
        if emb is None:
            return False, f"No face embedding found for unidentified face '{face_id}'"
        emb_arr = np.asarray(emb, dtype=np.float32)
        unid = self.face_history_db.get_unidentified_face(face_id)
        thumb = unid.get("thumbnail") if unid else None
        self.face_db.add_identity(name, emb_arr, thumbnail=thumb)
        self.face_history_db.enroll_unidentified_face(face_id, name)
        with self._state_lock:
            for ev in self.events:
                if ev.get("face_id") == face_id:
                    ev["type"] = "face"
                    ev["unidentified"] = False
                    ev["name"] = name
                    ev["message"] = f"Enrolled identity '{name}'"
        return True, f"Successfully enrolled '{name}'."

    def _cam_name(self, cam_id: str) -> str:
        for c in self.config.get("cameras", []) or []:
            if c.get("id") == cam_id:
                return c.get("name", cam_id)
        return cam_id

    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._state_lock:
            latest = dict(self._latest)
        face_rec_on = self.is_model_enabled("face_recognizer")
        return {
            "persons": sum(v["persons"] for v in latest.values()),
            "faces_recognized": sum(v["known_faces"] for v in latest.values()) if face_rec_on else 0,
            "plates": sum(v["plates"] for v in latest.values()),
            "total_fps": round(sum(self.runtime.camera_fps.values()), 1),
        }

    def get_runtime_metrics(self) -> Dict[str, Any]:
        return self.runtime.metrics()

    def get_models_status(self) -> List[Dict[str, Any]]:
        """Returns status and specs of all inference models / analytics."""
        plan = self.plan or self.compute_plan()
        backend = self.runtime.backend
        pose_model = plan.tier.pose_model
        pose_label = "YOLOv8s-Pose" if pose_model == "yolov8spose" else "YOLOv8n-Pose"
        cadence = f"{plan.fps} FPS/camera ({plan.tier.name})"
        res = "640x640" + (" + SAHI tiles" if plan.tier.tiling else "")
        running = self.is_running
        zone_count = len(self.config.get("zone_crossing", {}).get("zones", []) or [])

        def st(mid):
            return self.is_model_enabled(mid)

        return [
            {
                "id": "human_detector", "name": f"{pose_label} Person Detector", "task": "Object Detection",
                "description": "Person boxes from the shared YOLOv8-pose model (INT8 on Metis) + OC-SORT tracking",
                "enabled": st("human_detector"), "inferencing": running, "backend": backend,
                "weights": f"{pose_model} (INT8 PTQ .axm)", "input_size": [640, 640], "resolution": res,
                "conf_threshold": self.config.get("voyager", {}).get("thresholds", {}).get("person_conf", 0.4),
                "fps_cadence": cadence, "model_profile": self.performance_profile, "capacity": plan.to_dict(),
                "icon": "user-check", "category": "primary",
            },
            {
                "id": "pose_estimator", "name": f"{pose_label} Keypoints", "task": "Pose Estimation",
                "description": "17-keypoint skeletons from the same model pass (toggle controls overlay only)",
                "enabled": st("pose_estimator"), "inferencing": running, "backend": backend,
                "weights": f"{pose_model} (INT8 PTQ .axm)", "input_size": [640, 640],
                "fps_cadence": cadence, "icon": "activity", "category": "primary",
            },
            {
                "id": "face_recognizer", "name": "RetinaFace + FaceNet Recognizer", "task": "Biometric Identification",
                "description": "RetinaFace detection → face alignment → 512-d FaceNet embeddings on Metis",
                "enabled": st("face_recognizer"), "inferencing": running and st("face_recognizer"),
                "backend": backend, "weights": "retinaface-mb0.25 + facenet-vggface2 (INT8)",
                "input_size": [160, 160], "match_threshold": self.match_threshold,
                "fps_cadence": cadence, "icon": "smile", "category": "primary",
            },
            {
                "id": "license_plate_recognizer", "name": "License Plate Recognition", "task": "OCR / Identification",
                "description": "YOLOv8n plate detector → LPRNet OCR on Metis (zoo weights trained on CCPD plates)",
                "enabled": st("license_plate_recognizer"),
                "inferencing": running and st("license_plate_recognizer"), "backend": backend,
                "weights": "yolov8n-license-plate + lprnet (INT8)", "input_size": [640, 640],
                "fps_cadence": cadence, "icon": "hash", "category": "model_zoo",
            },
            {
                "id": "zone_crossing", "name": "Zone Crossing Detection", "task": "Analytics",
                "description": f"Line/polygon crossing on OC-SORT tracks — {zone_count} zone(s) configured",
                "enabled": st("zone_crossing"), "inferencing": running and st("zone_crossing"),
                "backend": "rule_based", "weights": "N/A (rule-based)", "zone_count": zone_count,
                "icon": "layers", "category": "analytics",
            },
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Plans capacity and starts the inference runtime."""
        self.is_running = True
        self.plan = None
        try:
            self.replan_now()
        except Exception:
            self.is_running = False
            raise
        self._replan_thread = threading.Thread(target=self._replan_worker, name="replan", daemon=True)
        self._replan_thread.start()
        print("[PIPELINE SUCCESS] System initialized and actively processing streams.")

    def stop(self):
        """Stops the processing pipeline."""
        self.is_running = False
        self._replan_evt.set()
        try:
            self.runtime.stop()
        finally:
            try:
                self.face_history_db.close()
            except Exception:
                pass
            self.preview.stop()
