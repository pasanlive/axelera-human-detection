"""
Pipeline: End-to-End Multi-Camera Detection, Pose, and Face Recognition System Orchestrator.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

from src.camera.stream_manager import StreamManager
from src.inference.yolo_detector import YOLODetector
from src.inference.pose_estimator import PoseEstimator
from src.inference.face_recognizer import FaceRecognizer
from src.utils.face_db import FaceDatabase
from src.utils.plate_db import PlateDatabase
from src.tracking.byte_tracker import ByteTracker
from src.utils.visualization import Visualizer


class MultiCameraPipeline:
    """Orchestrates multi-camera acquisition, Axelera Metis model inference, tracking, and visualization."""

    # Pose backends are mutually exclusive — only one may be active at a time
    POSE_MUTEX_GROUP = {"pose_estimator", "keypoint_detector"}

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        print("==========================================================")
        print("  Initializing Axelera Metis Multi-Camera Pipeline System ")
        print("==========================================================")

        # ── Performance settings (single parse, no duplication) ─────────────
        perf_cfg = config.get("performance", {})
        self.detect_interval = max(1, int(perf_cfg.get("detect_interval", 1)))
        self.pose_interval   = max(1, int(perf_cfg.get("pose_interval", 2)))
        self.face_interval   = max(1, int(perf_cfg.get("face_interval", 3)))

        # ── Frame counters & caches ──────────────────────────────────────────
        self.frame_counters: Dict[str, int] = {}
        self.cached_detections: Dict[str, List[Dict[str, Any]]] = {}
        self.cached_poses: Dict[str, List[Dict[str, Any]]] = {}
        self.face_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # ── Model enablement state ───────────────────────────────────────────
        models_cfg = config.get("models", {})
        self.enabled_models: Dict[str, bool] = {
            "human_detector":           models_cfg.get("human_detector",           {}).get("enabled", True),
            "pose_estimator":           models_cfg.get("pose_estimator",           {}).get("enabled", True),
            "face_recognizer":          models_cfg.get("face_recognizer",          {}).get("enabled", True),
            # Model zoo — disabled by default
            "yolo11n_detector":         models_cfg.get("yolo11n_detector",         {}).get("enabled", False),
            "yolov5n_detector":         models_cfg.get("yolov5n_detector",         {}).get("enabled", False),
            "keypoint_detector":        models_cfg.get("keypoint_detector",        {}).get("enabled", False),
            "license_plate_recognizer": models_cfg.get("license_plate_recognizer", {}).get("enabled", False),
            "zone_crossing":            config.get("zone_crossing",               {}).get("enabled", False),
        }
        self.camera_model_overrides: Dict[str, Dict[str, bool]] = {}

        # ── Face Database ────────────────────────────────────────────────────
        db_path = config.get("face_db", {}).get("path", "data/face_db.json")
        self.face_db = FaceDatabase(db_path=db_path)

        # ── Plate Database ───────────────────────────────────────────────────
        plate_db_path = config.get("plate_db", {}).get("path", "data/plate_db.json")
        self.plate_db = PlateDatabase(db_path=plate_db_path)

        # ── Primary models (lazy: only load if enabled) ──────────────────────
        hw_cfg = config.get("hardware", {})

        print("[PIPELINE] Initializing YOLO Human Detector...")
        self.detector = YOLODetector(models_cfg["human_detector"]) \
            if self.enabled_models["human_detector"] else None

        print("[PIPELINE] Initializing YOLO Pose Estimator...")
        self.pose_estimator = PoseEstimator(models_cfg["pose_estimator"]) \
            if self.enabled_models["pose_estimator"] else None

        print("[PIPELINE] Initializing Face Recognizer...")
        self.face_recognizer = FaceRecognizer(models_cfg["face_recognizer"], self.face_db) \
            if self.enabled_models["face_recognizer"] else None

        # ── Model zoo: lazily initialized on first enable ────────────────────
        # Detector alternatives
        self._yolo11n_detector: Optional[YOLODetector] = None
        self._yolov5n_detector: Optional[YOLODetector] = None
        # Pose alternative (mutually exclusive with pose_estimator)
        self._keypoint_detector = None
        # LPR
        self._lpr: Optional[Any] = None
        # Zone crossing (rule-based — no model)
        self._zone_crossing: Optional[Any] = None

        # Eagerly initialize model zoo items that are already enabled in config
        if self.enabled_models["yolo11n_detector"]:
            self._ensure_yolo11n_detector()
        if self.enabled_models["yolov5n_detector"]:
            self._ensure_yolov5n_detector()
        if self.enabled_models["keypoint_detector"]:
            self._ensure_keypoint_detector()
        if self.enabled_models["license_plate_recognizer"]:
            self._ensure_lpr()
        if self.enabled_models["zone_crossing"]:
            self._ensure_zone_crossing()

        # ── Trackers per camera stream ────────────────────────────────────────
        self.trackers: Dict[str, ByteTracker] = {}

        # ── Multi-Camera Stream Manager ───────────────────────────────────────
        self.stream_manager = StreamManager(config.get("cameras", []))
        self.stream_manager.initialize_cameras()

        # ── Visualizer ────────────────────────────────────────────────────────
        self.visualizer = Visualizer(config)

        self.is_running = False

    # ─────────────────────────────────────────────────────────────────────────
    # Lazy initializers for model zoo components
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_yolo11n_detector(self):
        if self._yolo11n_detector is None:
            cfg = self.config.get("models", {}).get("yolo11n_detector", {})
            if not cfg:
                print("[PIPELINE] yolo11n_detector config not found.")
                return
            print("[PIPELINE] Lazy-loading YOLO11n Detector...")
            self._yolo11n_detector = YOLODetector(cfg)

    def _ensure_yolov5n_detector(self):
        if self._yolov5n_detector is None:
            cfg = self.config.get("models", {}).get("yolov5n_detector", {})
            if not cfg:
                print("[PIPELINE] yolov5n_detector config not found.")
                return
            print("[PIPELINE] Lazy-loading YOLOv5n Detector...")
            self._yolov5n_detector = YOLODetector(cfg)

    def _ensure_keypoint_detector(self):
        if self._keypoint_detector is None:
            cfg = self.config.get("models", {}).get("keypoint_detector", {})
            if not cfg:
                print("[PIPELINE] keypoint_detector config not found.")
                return
            print("[PIPELINE] Lazy-loading YOLO11n-Pose Keypoint Detector...")
            from src.inference.keypoint_detector import KeypointDetector
            self._keypoint_detector = KeypointDetector(cfg)

    def _ensure_lpr(self):
        if self._lpr is None:
            cfg = self.config.get("models", {}).get("license_plate_recognizer", {})
            if not cfg:
                print("[PIPELINE] license_plate_recognizer config not found.")
                return
            print("[PIPELINE] Lazy-loading License Plate Recognizer...")
            from src.inference.license_plate_recognizer import LicensePlateRecognizer
            self._lpr = LicensePlateRecognizer(cfg, self.plate_db)

    def _ensure_zone_crossing(self):
        if self._zone_crossing is None:
            zone_cfg = self.config.get("zone_crossing", {})
            zones = zone_cfg.get("zones", [])
            print(f"[PIPELINE] Lazy-loading Zone Crossing Detector ({len(zones)} zone(s))...")
            from src.analytics.zone_crossing import ZoneCrossingDetector
            self._zone_crossing = ZoneCrossingDetector(zones)

    # ─────────────────────────────────────────────────────────────────────────
    # Active detector / pose backend helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_active_detector(self):
        """Returns the active human detector (model zoo overrides primary if enabled)."""
        if self.enabled_models.get("yolo11n_detector") and self._yolo11n_detector:
            return self._yolo11n_detector
        if self.enabled_models.get("yolov5n_detector") and self._yolov5n_detector:
            return self._yolov5n_detector
        return self.detector  # primary YOLOv8n

    def _get_active_pose_backend(self):
        """Returns the active pose backend. keypoint_detector takes priority if enabled."""
        if self.enabled_models.get("keypoint_detector") and self._keypoint_detector:
            return self._keypoint_detector
        if self.enabled_models.get("pose_estimator") and self.pose_estimator:
            return self.pose_estimator
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Model enable/disable with pose mutex
    # ─────────────────────────────────────────────────────────────────────────

    def is_model_enabled(self, model_name: str, cam_id: Any = None) -> bool:
        """Checks whether a specific model is enabled globally or for a specific camera."""
        if cam_id and cam_id in self.camera_model_overrides:
            cam_overrides = self.camera_model_overrides[cam_id]
            if model_name in cam_overrides:
                return cam_overrides[model_name]
        return self.enabled_models.get(model_name, True)

    def set_model_enabled(self, model_name: str, enabled: bool, cam_id: Any = None):
        """
        Enables or disables model inference globally or for a specific camera.
        Enforces the pose mutex: enabling one pose backend auto-disables the other.
        Lazy-initializes model zoo components on first enable.
        """
        if cam_id and cam_id != "all":
            if cam_id not in self.camera_model_overrides:
                self.camera_model_overrides[cam_id] = {}
            self.camera_model_overrides[cam_id][model_name] = enabled
            print(f"[PIPELINE ADMIN] Model '{model_name}' on stream '{cam_id}' set to: {'ENABLED' if enabled else 'DISABLED'}")
        else:
            # Pose mutex: enabling one pose backend disables the other
            if enabled and model_name in self.POSE_MUTEX_GROUP:
                for other in self.POSE_MUTEX_GROUP:
                    if other != model_name and self.enabled_models.get(other):
                        self.enabled_models[other] = False
                        for c_id in self.camera_model_overrides:
                            if other in self.camera_model_overrides[c_id]:
                                del self.camera_model_overrides[c_id][other]
                        self._clear_model_cache(other)
                        print(f"[PIPELINE ADMIN] Pose mutex: '{other}' auto-disabled.")

            self.enabled_models[model_name] = enabled
            for c_id in self.camera_model_overrides:
                if model_name in self.camera_model_overrides[c_id]:
                    del self.camera_model_overrides[c_id][model_name]
            print(f"[PIPELINE ADMIN] Model '{model_name}' globally set to: {'ENABLED' if enabled else 'DISABLED'}")

        # Lazy-load on first enable
        if enabled:
            if model_name == "yolo11n_detector":
                self._ensure_yolo11n_detector()
            elif model_name == "yolov5n_detector":
                self._ensure_yolov5n_detector()
            elif model_name == "keypoint_detector":
                self._ensure_keypoint_detector()
            elif model_name == "license_plate_recognizer":
                self._ensure_lpr()
            elif model_name == "zone_crossing":
                self._ensure_zone_crossing()
            elif model_name == "human_detector" and self.detector is None:
                cfg = self.config.get("models", {}).get("human_detector", {})
                if cfg:
                    print("[PIPELINE] Lazy-loading primary YOLO Human Detector...")
                    self.detector = YOLODetector(cfg)
            elif model_name == "pose_estimator" and self.pose_estimator is None:
                cfg = self.config.get("models", {}).get("pose_estimator", {})
                if cfg:
                    print("[PIPELINE] Lazy-loading primary YOLO Pose Estimator...")
                    self.pose_estimator = PoseEstimator(cfg)
            elif model_name == "face_recognizer" and self.face_recognizer is None:
                cfg = self.config.get("models", {}).get("face_recognizer", {})
                if cfg:
                    print("[PIPELINE] Lazy-loading Face Recognizer...")
                    self.face_recognizer = FaceRecognizer(cfg, self.face_db)

        # Clear stale cache on disable
        if not enabled:
            self._clear_model_cache(model_name, cam_id if (cam_id and cam_id != "all") else None)

    def _clear_model_cache(self, model_name: str, cam_id: Optional[str] = None):
        """Clears cached inference data when a model is disabled."""
        def _clear_dict(d: dict):
            if cam_id:
                if cam_id in d:
                    d[cam_id] = [] if isinstance(d.get(cam_id), list) else {}
            else:
                for k in d:
                    d[k] = [] if isinstance(d.get(k), list) else {}

        if model_name in ("human_detector", "yolo11n_detector", "yolov5n_detector"):
            _clear_dict(self.cached_detections)
        elif model_name in ("pose_estimator", "keypoint_detector"):
            _clear_dict(self.cached_poses)
        elif model_name == "face_recognizer":
            _clear_dict(self.face_cache)

    def reload_detector_model(self, model_profile: str, model_name: str,
                               input_size: List[int],
                               axm_path: Optional[str] = None,
                               onnx_path: Optional[str] = None,
                               detect_interval: Optional[int] = None):
        """Hot-reloads human detection model configuration."""
        det_cfg = self.config.setdefault("models", {}).setdefault("human_detector", {})
        det_cfg["model_profile"] = model_profile
        det_cfg["model_name"] = model_name
        det_cfg["input_size"] = list(input_size)
        if axm_path is not None:
            det_cfg["axm_path"] = axm_path
        if onnx_path is not None:
            det_cfg["onnx_path"] = onnx_path
        if detect_interval is not None:
            self.detect_interval = max(1, int(detect_interval))
            self.config.setdefault("performance", {})["detect_interval"] = self.detect_interval

        self.detector = YOLODetector(det_cfg)
        for cid in self.cached_detections:
            self.cached_detections[cid] = []
        print(f"[PIPELINE ADMIN] Detector reloaded: profile={model_profile}, model={model_name}, size={input_size}")

    def get_models_status(self) -> List[Dict[str, Any]]:
        """Returns status, specs, and telemetry of all registered AI inference models."""
        models_cfg = self.config.get("models", {})
        det_cfg   = models_cfg.get("human_detector", {})
        pose_cfg  = models_cfg.get("pose_estimator", {})
        face_cfg  = models_cfg.get("face_recognizer", {})
        kpt_cfg   = models_cfg.get("keypoint_detector", {})
        y11_cfg   = models_cfg.get("yolo11n_detector", {})
        y5n_cfg   = models_cfg.get("yolov5n_detector", {})
        lpr_cfg   = models_cfg.get("license_plate_recognizer", {})

        def _backend(obj, attr="engine"):
            return getattr(getattr(obj, attr, None), "backend", "Unknown") if obj else "Not Loaded"

        det_en   = self.is_model_enabled("human_detector")
        pose_en  = self.is_model_enabled("pose_estimator")
        face_en  = self.is_model_enabled("face_recognizer")
        kpt_en   = self.is_model_enabled("keypoint_detector")
        y11_en   = self.is_model_enabled("yolo11n_detector")
        y5n_en   = self.is_model_enabled("yolov5n_detector")
        lpr_en   = self.is_model_enabled("license_plate_recognizer")
        zone_en  = self.is_model_enabled("zone_crossing")

        model_name_str = str(det_cfg.get("model_name", "yolov8n.pt")).lower()
        det_display = "YOLO11 Ultra-Light Detector" if "yolo11" in model_name_str else "YOLOv8 Human Detector"
        det_cadence = "Every frame (1:1)" if self.detect_interval == 1 else f"Every {self.detect_interval} frames"

        zone_zone_count = len(self._zone_crossing.get_zones_for_camera(None) if False else []) \
            if self._zone_crossing else 0
        # Count total configured zones from config
        zone_count = len(self.config.get("zone_crossing", {}).get("zones", []))

        return [
            {
                "id": "human_detector",
                "name": det_display,
                "task": "Object Detection",
                "description": "Primary human detector — bounding box localization & confidence scoring",
                "enabled": det_en,
                "inferencing": det_en and self.is_running,
                "backend": _backend(self.detector),
                "weights": det_cfg.get("axm_path") or det_cfg.get("model_name", "yolov8n.pt"),
                "model_profile": det_cfg.get("model_profile", "ultra_light"),
                "input_size": det_cfg.get("input_size", [320, 320]),
                "conf_threshold": det_cfg.get("conf_threshold", 0.45),
                "fps_cadence": det_cadence,
                "detect_interval": self.detect_interval,
                "icon": "user-check",
                "category": "primary"
            },
            {
                "id": "pose_estimator",
                "name": "YOLOv8-Pose Estimator",
                "task": "Pose Estimation",
                "description": "17-keypoint skeletal joint tracking (YOLOv8n-Pose)",
                "enabled": pose_en,
                "inferencing": pose_en and self.is_running,
                "backend": _backend(self.pose_estimator),
                "weights": pose_cfg.get("axm_path") or pose_cfg.get("model_name", "yolov8n-pose.pt"),
                "input_size": pose_cfg.get("input_size", [512, 512]),
                "conf_threshold": pose_cfg.get("conf_threshold", 0.50),
                "fps_cadence": f"Every {self.pose_interval} frames",
                "icon": "activity",
                "category": "primary",
                "mutex_group": "pose"
            },
            {
                "id": "face_recognizer",
                "name": "ArcFace Facial Recognizer",
                "task": "Biometric Identification",
                "description": "512-d facial embedding extraction and identity matching",
                "enabled": face_en,
                "inferencing": face_en and self.is_running,
                "backend": _backend(self.face_recognizer, "embedder_engine"),
                "weights": face_cfg.get("embedder_axm") or face_cfg.get("embedder_onnx", "arcface_mobilefacenet.onnx"),
                "input_size": face_cfg.get("input_size", [112, 112]),
                "match_threshold": face_cfg.get("match_threshold", 0.60),
                "fps_cadence": f"Every {self.face_interval} frames",
                "icon": "smile",
                "category": "primary"
            },
            # ── Model Zoo ──────────────────────────────────────────────────────
            {
                "id": "yolo11n_detector",
                "name": "YOLO11n Ultra-Light Detector",
                "task": "Object Detection",
                "description": "YOLO11n (C2PSA attention) @ 320×320 — model zoo alternative detector",
                "enabled": y11_en,
                "inferencing": y11_en and self.is_running,
                "backend": _backend(self._yolo11n_detector),
                "weights": y11_cfg.get("onnx_path", "models/onnx/yolo11n_320.onnx"),
                "model_profile": "yolo11_ultra_light",
                "input_size": y11_cfg.get("input_size", [320, 320]),
                "conf_threshold": y11_cfg.get("conf_threshold", 0.45),
                "icon": "cpu",
                "category": "model_zoo"
            },
            {
                "id": "yolov5n_detector",
                "name": "YOLOv5n Extreme-Light Detector",
                "task": "Object Detection",
                "description": "YOLOv5n @ 320×320 — 1.8M params, lowest compute footprint",
                "enabled": y5n_en,
                "inferencing": y5n_en and self.is_running,
                "backend": _backend(self._yolov5n_detector),
                "weights": y5n_cfg.get("onnx_path", "models/onnx/yolov5n_320.onnx"),
                "model_profile": "extreme_light",
                "input_size": y5n_cfg.get("input_size", [320, 320]),
                "conf_threshold": y5n_cfg.get("conf_threshold", 0.45),
                "icon": "zap",
                "category": "model_zoo"
            },
            {
                "id": "keypoint_detector",
                "name": "YOLO11n-Pose Keypoint Detector",
                "task": "Pose Estimation",
                "description": "YOLO11n-Pose @ 512×512 — model zoo alternative (mutually exclusive with YOLOv8-Pose)",
                "enabled": kpt_en,
                "inferencing": kpt_en and self.is_running,
                "backend": _backend(self._keypoint_detector),
                "weights": kpt_cfg.get("onnx_path", "models/onnx/yolo11n-pose_512.onnx"),
                "input_size": kpt_cfg.get("input_size", [512, 512]),
                "conf_threshold": kpt_cfg.get("conf_threshold", 0.50),
                "fps_cadence": f"Every {self.pose_interval} frames",
                "icon": "git-branch",
                "category": "model_zoo",
                "mutex_group": "pose"
            },
            {
                "id": "license_plate_recognizer",
                "name": "License Plate Recognition",
                "task": "OCR / Identification",
                "description": "YOLOv8n plate detector + CRNN OCR — reads and logs vehicle license plates",
                "enabled": lpr_en,
                "inferencing": lpr_en and self.is_running,
                "backend": _backend(self._lpr, "detector_engine") if self._lpr else "Not Loaded",
                "weights": lpr_cfg.get("detector_onnx", "models/onnx/yolov8n-plate.onnx"),
                "input_size": lpr_cfg.get("detector_input_size", [640, 640]),
                "conf_threshold": lpr_cfg.get("conf_threshold", 0.45),
                "icon": "hash",
                "category": "model_zoo"
            },
            {
                "id": "zone_crossing",
                "name": "Zone Crossing Detection",
                "task": "Analytics",
                "description": f"Virtual line/polygon zone crossing detection — {zone_count} zone(s) configured",
                "enabled": zone_en,
                "inferencing": zone_en and self.is_running,
                "backend": "rule_based",
                "weights": "N/A (rule-based)",
                "zone_count": zone_count,
                "icon": "layers",
                "category": "analytics"
            },
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Starts multi-camera capture and processing loop."""
        self.stream_manager.start()
        self.is_running = True

        for cam_id in self.stream_manager.cameras.keys():
            self.trackers[cam_id] = ByteTracker(
                track_thresh=self.config.get("tracking", {}).get("track_thresh", 0.5)
            )
            self.frame_counters[cam_id] = 0
            self.cached_detections[cam_id] = []
            self.cached_poses[cam_id] = []
            self.face_cache[cam_id] = {}

        print("[PIPELINE SUCCESS] System initialized and actively processing streams.")

    def process_step(self) -> Dict[str, np.ndarray]:
        """
        Executes one processing step across all active camera streams.
        Only models that are enabled perform inference.
        :return: Dict of camera_id -> rendered frame numpy array
        """
        frames = self.stream_manager.get_all_frames()
        output_frames = {}

        for cam_id, frame in frames.items():
            cam_obj = self.stream_manager.cameras.get(cam_id)
            cam_name = cam_obj.name if cam_obj else cam_id
            fps = cam_obj.fps if cam_obj else 0.0

            self.frame_counters[cam_id] = self.frame_counters.get(cam_id, 0) + 1
            curr_frame_idx = self.frame_counters[cam_id]

            # ── Step A: Human Detection ───────────────────────────────────────
            active_detector = self._get_active_detector()
            det_model_en = (
                self.is_model_enabled("human_detector", cam_id) or
                self.is_model_enabled("yolo11n_detector", cam_id) or
                self.is_model_enabled("yolov5n_detector", cam_id)
            ) and active_detector is not None

            if det_model_en:
                should_detect = (curr_frame_idx % self.detect_interval == 0) or not self.cached_detections.get(cam_id)
                if should_detect:
                    raw_detections = active_detector.detect(frame)
                    if cam_id in self.trackers:
                        detections = self.trackers[cam_id].update(raw_detections) if raw_detections else []
                    else:
                        detections = raw_detections
                    self.cached_detections[cam_id] = detections
                else:
                    if cam_id in self.trackers:
                        active_tracks = self.trackers[cam_id].get_active_tracks()
                        detections = active_tracks if active_tracks else self.cached_detections.get(cam_id, [])
                    else:
                        detections = self.cached_detections.get(cam_id, [])
            else:
                detections = []
                self.cached_detections[cam_id] = []

            # ── Step B: Pose Estimation (active backend — poses mutex enforced) ──
            pose_backend = self._get_active_pose_backend()
            if pose_backend is not None and (
                self.is_model_enabled("pose_estimator", cam_id) or
                self.is_model_enabled("keypoint_detector", cam_id)
            ):
                if curr_frame_idx % self.pose_interval == 0 or not self.cached_poses.get(cam_id):
                    poses = pose_backend.estimate_pose(frame)
                    self.cached_poses[cam_id] = poses
                else:
                    poses = self.cached_poses.get(cam_id, [])
            else:
                poses = []
                self.cached_poses[cam_id] = []

            # ── Step C: Face Recognition ──────────────────────────────────────
            active_faces = []
            if self.is_model_enabled("face_recognizer", cam_id) and self.face_recognizer:
                if self.is_model_enabled("human_detector", cam_id) or \
                   self.is_model_enabled("yolo11n_detector", cam_id) or \
                   self.is_model_enabled("yolov5n_detector", cam_id):
                    person_boxes = []
                    uncached_person_boxes = []
                    for d in detections:
                        bbox = d["bbox"]
                        track_id = d.get("track_id")
                        person_boxes.append(bbox)
                        if track_id is None or track_id not in self.face_cache.get(cam_id, {}):
                            uncached_person_boxes.append(bbox)

                    if curr_frame_idx % self.face_interval == 0 or uncached_person_boxes:
                        target_boxes = uncached_person_boxes if (curr_frame_idx % self.face_interval != 0) else person_boxes
                        if target_boxes:
                            new_faces = self.face_recognizer.recognize_faces_in_frame(frame, person_bboxes=target_boxes)
                            for face in new_faces:
                                f_box = face["bbox"]
                                for d in detections:
                                    if self._bbox_overlap(f_box, d["bbox"]) > 0.3:
                                        tid = d.get("track_id")
                                        if tid:
                                            self.face_cache[cam_id][tid] = face

                    for d in detections:
                        tid = d.get("track_id")
                        if tid and tid in self.face_cache.get(cam_id, {}):
                            face_info = self.face_cache[cam_id][tid].copy()
                            x1, y1, x2, y2 = d["bbox"]
                            head_h = int((y2 - y1) * 0.35)
                            face_info["bbox"] = [x1, y1, x2, y1 + head_h]
                            active_faces.append(face_info)
                else:
                    if curr_frame_idx % self.face_interval == 0:
                        active_faces = self.face_recognizer.recognize_faces_in_frame(frame, person_bboxes=None)
            else:
                self.face_cache[cam_id] = {}

            # ── Step D: License Plate Recognition ────────────────────────────
            plates = []
            if self.is_model_enabled("license_plate_recognizer", cam_id) and self._lpr:
                person_bboxes = [d["bbox"] for d in detections] if detections else None
                plates = self._lpr.detect_plates(frame, person_bboxes=person_bboxes)
                # Log to plate DB
                for plate in plates:
                    if plate.get("plate_text"):
                        self.plate_db.log_plate(
                            plate["plate_text"], cam_id,
                            plate.get("confidence", 0.0),
                            plate.get("ocr_confidence", 0.0)
                        )

            # ── Step E: Zone Crossing ──────────────────────────────────────────
            zone_events = []
            if self.is_model_enabled("zone_crossing", cam_id) and self._zone_crossing:
                if self._zone_crossing.has_zones_for_camera(cam_id) and detections:
                    active_track_ids = []
                    for d in detections:
                        tid = d.get("track_id")
                        if tid is not None:
                            active_track_ids.append(tid)
                            bbox = d["bbox"]
                            centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                            events = self._zone_crossing.update(cam_id, tid, centroid)
                            zone_events.extend(events)
                    self._zone_crossing.purge_stale_tracks(cam_id, active_track_ids)

            # ── Step F: Visualization ──────────────────────────────────────────
            zone_defs = []
            if self.is_model_enabled("zone_crossing", cam_id) and self._zone_crossing:
                zone_defs = self._zone_crossing.get_zones_for_camera(cam_id)

            rendered = self.visualizer.draw_frame(
                frame=frame,
                detections=detections,
                poses=poses,
                faces=active_faces,
                plates=plates,
                zones=zone_defs,
                zone_counts=self._zone_crossing.get_zone_counts(cam_id) if self._zone_crossing else {},
                stream_title=f"{cam_name}",
                fps=fps
            )

            output_frames[cam_id] = rendered

        return output_frames

    @staticmethod
    def _bbox_overlap(b1: List[float], b2: List[float]) -> float:
        """Calculates area overlap ratio between face box and person box."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        return inter / area1 if area1 > 0 else 0.0

    def compose_grid(self, frames_dict: Dict[str, np.ndarray], grid_w: int = 1280, grid_h: int = 720) -> np.ndarray:
        """Combines multiple camera outputs into a single tiled grid layout."""
        if not frames_dict:
            blank = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            cv2.putText(blank, "No Active Camera Streams", (grid_w // 2 - 200, grid_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank

        frames_list = list(frames_dict.values())
        num_streams = len(frames_list)

        if num_streams == 1:
            return cv2.resize(frames_list[0], (grid_w, grid_h))

        cols = 2 if num_streams <= 4 else 3
        rows = int(np.ceil(num_streams / cols))
        tile_w = grid_w // cols
        tile_h = grid_h // rows

        grid = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
        for idx, frame in enumerate(frames_list):
            r = idx // cols
            c = idx % cols
            tile = cv2.resize(frame, (tile_w, tile_h))
            grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile

        return cv2.resize(grid, (grid_w, grid_h))

    def stop(self):
        """Stops the processing pipeline."""
        self.is_running = False
        self.stream_manager.stop()
