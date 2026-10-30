"""
Network builder: composes Voyager SDK 1.6 network YAMLs from Model Zoo building blocks.

Every network the application can run is a *combination* of:

* a person/pose model (one model does detection AND 17-keypoint pose, so there
  is no separate detector)                               -> yolov8npose | yolov8spose
* an OC-SORT tracker (runs on the host inside the GStreamer pipeline)
* optional face branch: RetinaFace-mb0.25 -> (face align) -> FaceNet embeddings
* optional plate branch: YOLOv8n licence-plate detector -> LPRNet OCR

All models are standard Axelera Model Zoo topologies, so they are quantized
(INT8 PTQ) and compiled by ``deploy.py`` and mapped fully onto the Metis AIPU.
Decoding/NMS runs in Voyager's optimised C++ decoders - no Python post-processing.

The network name encodes the composition, e.g. ``upside-yolov8spose-face-lpr``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

NETWORK_PREFIX = "upside"
POSE_MODELS = ("yolov8npose", "yolov8spose")
FEATURES = ("face", "lpr")

# Task names exposed in FrameResult.meta - the result parser depends on these.
TASK_PERSONS = "persons"
TASK_TRACKS = "person_tracks"
TASK_FACES = "faces"
TASK_FACE_EMB = "face_embeddings"
TASK_PLATES = "plates"
TASK_PLATE_TEXT = "plate_text"

_FW = "$AXELERA_FRAMEWORK"

# ── Model Zoo definitions (copied from Voyager SDK v1.6.0 zoo YAMLs) ──────────────
_MODEL_DEFS: Dict[str, Dict[str, Any]] = {
    "yolov8npose": {
        "name": "yolov8npose-coco-onnx",
        "def": {
            "class": "AxONNXModel",
            "class_path": f"{_FW}/ax_models/base_onnx.py",
            "weight_path": "weights/yolov8n-pose-v8.1.onnx",
            "weight_url": "https://media.axelera.ai/artifacts/model_cards/weights/yolo/keypoint_detection/yolov8n-pose-v8.1.onnx",
            "weight_md5": "67a56731f44db74079357437d37cb76d",
            "task_category": "KeypointDetection",
            "input_tensor_layout": "NCHW",
            "input_tensor_shape": [1, 3, 640, 640],
            "input_color_format": "RGB",
            "num_classes": 1,
            "dataset": "CocoDataset-keypoint-COCO2017",
        },
    },
    "yolov8spose": {
        "name": "yolov8spose-coco-onnx",
        "def": {
            "class": "AxONNXModel",
            "class_path": f"{_FW}/ax_models/base_onnx.py",
            "weight_path": "weights/yolov8s-pose-v8.1.onnx",
            "weight_url": "https://media.axelera.ai/artifacts/model_cards/weights/yolo/keypoint_detection/yolov8s-pose-v8.1.onnx",
            "weight_md5": "9542fbf0bbc6e44de41d1034ec1c9b4c",
            "task_category": "KeypointDetection",
            "input_tensor_layout": "NCHW",
            "input_tensor_shape": [1, 3, 640, 640],
            "input_color_format": "RGB",
            "num_classes": 1,
            "dataset": "CocoDataset-keypoint-COCO2017",
        },
    },
    "retinaface": {
        "name": "retinaface-mobilenet0.25-widerface-onnx",
        "def": {
            "class": "AxONNXModel",
            "class_path": f"{_FW}/ax_models/base_onnx.py",
            "weight_path": "weights/Retinaface_mb0.25_pytorch.onnx",
            "weight_url": "https://media.axelera.ai/artifacts/model_cards/weights/others/object_detection/Retinaface_mb0.25_pytorch.onnx",
            "weight_md5": "9232a08b7056937f63ba26b6a3606546",
            "task_category": "ObjectDetection",
            "input_tensor_layout": "NCHW",
            "input_tensor_shape": [1, 3, 640, 640],
            "input_color_format": "BGR",
            "num_classes": 1,
            "dataset": "WiderFace",
            "extra_kwargs": {
                "RetinaFace": {
                    "cfg": {
                        "min_sizes": [[16, 32], [64, 128], [256, 512]],
                        "steps": [8, 16, 32],
                        "variance": [0.1, 0.2],
                        "clip": False,
                    }
                }
            },
        },
    },
    "facenet": {
        "name": "facenet-vggface2",
        "def": {
            "class": "AxFaceNet",
            "class_path": f"{_FW}/ax_models/torch/facenet.py",
            "weight_path": "weights/facenet_vgg_face_2.pt",
            "weight_url": "https://media.axelera.ai/artifacts/model_cards/weights/face_recognition/facenet_vgg_face_2.pt",
            "weight_md5": "13e767ebf459ce7459f130c3c1398a7b",
            # ReIdentification + decodeembeddings -> raw 512-d embeddings (EmbeddingsMeta),
            # matched against our own FaceDatabase on the host.
            "task_category": "ReIdentification",
            "input_tensor_layout": "NCHW",
            "input_tensor_shape": [1, 3, 160, 160],
            "input_color_format": "RGB",
            "num_classes": 8631,
            "dataset": "LFWTorchvisionPeople",
        },
    },
    "plate_detector": {
        "name": "yolov8n-license-plate",
        "def": {
            "class": "AxONNXModel",
            "class_path": f"{_FW}/ax_models/base_onnx.py",
            "weight_path": "weights/yolov8n_license_plate_detector.onnx",
            "weight_url": "https://media.axelera.ai/artifacts/weights/yolo/yolov8n_license_plate_detector.onnx",
            "weight_md5": "80df1cedb4bbaded6cba64d4a48d1ecc",
            "task_category": "ObjectDetection",
            "input_tensor_layout": "NCHW",
            "input_tensor_shape": [1, 3, 640, 640],
            "input_color_format": "RGB",
            "num_classes": 1,
            "dataset": "CocoDataset-LicensePlate",
        },
    },
    "lprnet": {
        "name": "lprnet",
        "def": {
            "class": "LPRNetTorchModel",
            "class_path": f"{_FW}/ax_models/torch/LPRNet.py",
            "weight_path": "weights/Final_LPRNet_model.pth",
            "weight_url": "https://media.axelera.ai/artifacts/model_cards/app_weights/license_plate_recog/Final_LPRNet_model.pth",
            "weight_md5": "1790d7e41cb14a18c051b181f8774708",
            "task_category": "LicensePlateRecognition",
            "input_tensor_layout": "NCHW",
            "input_tensor_shape": [1, 3, 24, 94],
            "input_color_format": "RGB",
            "num_classes": 68,
            "dataset": "LPRNetDataset",
            "extra_kwargs": {"trimmed_labels": False, "LPRNet": {"lpr_max_len": 8}},
        },
    },
}

_DATASETS: Dict[str, Dict[str, Any]] = {
    "CocoDataset-keypoint-COCO2017": {
        "class": "KptDataAdapter",
        "class_path": f"{_FW}/ax_datasets/objdataadapter.py",
        "data_dir_name": "coco",
        "label_type": "COCO2017",
    },
    "WiderFace": {
        "class": "WiderFaceDataAdapter",
        "class_path": f"{_FW}/ax_datasets/widerface.py",
        "data_dir_name": "widerface",
        "labels_path": f"{_FW}/ax_datasets/labels/face.names",
        "val_data": "val/label.txt",
    },
    "LFWTorchvisionPeople": {
        "class": "TorchvisionDataAdapter",
        "class_path": f"{_FW}/ax_datasets/torchvision.py",
        "dataset_name": "LFWPeople",
        "data_dir_name": "LFW",
        "image_set": "deepfunneled",
    },
    "CocoDataset-LicensePlate": {
        "class": "ObjDataAdapter",
        "class_path": f"{_FW}/ax_datasets/objdataadapter.py",
        "data_dir_name": "licenseplate_v4_resized640_aug3x-ACCURATE",
        "dataset_url": "https://media.axelera.ai/artifacts/data/licenseplate_v4_resized640_aug3x-ACCURATE.zip",
        "dataset_md5": "85a19e378bf699c0e263e79e4c1a4622",
        "dataset_drop_dirs": 0,
        "label_type": "YOLOv8",
        "ultralytics_data_yaml": "data.yaml",
    },
    "LPRNetDataset": {
        "class": "LPRNetDataAdapter",
        "class_path": f"{_FW}/ax_datasets/license_plate_recog.py",
        "data_dir_name": "license_plate_recog",
        "labels_path": f"{_FW}/ax_datasets/labels/lprnet-char.yaml",
    },
}

# Datasets whose calibration should use frames captured from *our* cameras.
# (FaceNet / LPRNet consume crops, so their zoo representative images are kept.)
_CAMERA_CALIBRATED_DATASETS = ("CocoDataset-keypoint-COCO2017", "WiderFace", "CocoDataset-LicensePlate")


@dataclass(frozen=True)
class NetworkSpec:
    """Identity of a deployable network."""

    pose_model: str = "yolov8npose"
    face: bool = False
    lpr: bool = False

    def __post_init__(self):
        if self.pose_model not in POSE_MODELS:
            raise ValueError(f"Unknown pose model {self.pose_model!r}; expected one of {POSE_MODELS}")

    @property
    def name(self) -> str:
        if self.pose_model == "yolov5m-v7-coco-tracker":
            return "yolov5m-v7-coco-tracker"
        parts = [NETWORK_PREFIX, self.pose_model]
        if self.face:
            parts.append("face")
        if self.lpr:
            parts.append("lpr")
        return "-".join(parts)

    @property
    def features(self) -> frozenset:
        return frozenset(f for f, on in (("face", self.face), ("lpr", self.lpr)) if on)

    @classmethod
    def from_name(cls, name: str) -> "NetworkSpec":
        if name == "yolov5m-v7-coco-tracker":
            return cls(pose_model="yolov5m-v7-coco-tracker", face=False, lpr=False)
        parts = name.split("-")
        if len(parts) < 2 or parts[0] != NETWORK_PREFIX:
            raise ValueError(f"Not an application network name: {name!r}")
        return cls(pose_model=parts[1], face="face" in parts[2:], lpr="lpr" in parts[2:])


@dataclass
class BuildOptions:
    """Tunables baked into the network YAML (thresholds, ROI limits, calibration)."""

    person_conf: float = 0.40
    person_nms_iou: float = 0.45
    face_conf: float = 0.50
    max_faces_per_frame: int = 6
    min_face_px: int = 32
    plate_conf: float = 0.40
    max_plates_per_frame: int = 4
    tracker_max_age: int = 50
    tracker_min_hits: int = 3
    tracker_iou: float = 0.3
    calibration_dir: Optional[str] = None
    """Absolute path to representative camera frames for INT8 PTQ (>=100 images recommended)."""
    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-model YAML overrides, e.g. {"yolov8spose": {"weight_path": "/abs/custom.onnx", "weight_url": None}}."""


def _allocate_cores(model_keys: List[str]) -> Dict[str, int]:
    """Split the 4 Metis AIPU cores between the models of a network.

    The person/pose model dominates the load (it runs on every frame), so it gets
    the lion's share. The secondary models get at least one core each.
    """
    if len(model_keys) == 1:
        return {model_keys[0]: 4}
    secondary = [k for k in model_keys if not k.endswith("pose")]
    pose = [k for k in model_keys if k.endswith("pose")][0]
    if len(secondary) >= 4:
        # 5 models on 4 cores: the SDK time-multiplexes; give each secondary model 1 core.
        alloc = {k: 1 for k in secondary}
        alloc[pose] = 1
        return alloc
    alloc = {k: 1 for k in secondary}
    alloc[pose] = 4 - len(secondary)
    return alloc


def build_network(spec: NetworkSpec, opts: Optional[BuildOptions] = None) -> Dict[str, Any]:
    """Returns the network YAML as a dict for the given composition."""
    opts = opts or BuildOptions()
    model_keys = [spec.pose_model]
    if spec.face:
        model_keys += ["retinaface", "facenet"]
    if spec.lpr:
        model_keys += ["plate_detector", "lprnet"]
    cores = _allocate_cores(model_keys)

    pose_name = _MODEL_DEFS[spec.pose_model]["name"]
    pipeline: List[Dict[str, Any]] = [
        {
            TASK_PERSONS: {
                "model_name": pose_name,
                "input": {"type": "image"},
                "preprocess": [
                    {"letterbox": {"width": 640, "height": 640, "scaleup": True}},
                    {"torch-totensor": None},
                ],
                "inference": {"handle_all": False},
                "postprocess": [
                    {
                        "decodeyolopose": {
                            "box_format": "xywh",
                            "normalized_coord": False,
                            "max_nms_boxes": 30000,
                            "conf_threshold": opts.person_conf,
                            "nms_iou_threshold": opts.person_nms_iou,
                            "nms_top_k": 300,
                        }
                    }
                ],
            }
        },
        {
            TASK_TRACKS: {
                "model_name": "tracker",
                "input": {"source": "full", "color_format": "RGB"},
                "cv_process": [
                    {
                        "tracker": {
                            "algorithm": "oc-sort",
                            "bbox_task_name": TASK_PERSONS,
                            "history_length": 30,
                            "algo_params": {
                                "max_age": opts.tracker_max_age,
                                "min_hits": opts.tracker_min_hits,
                                "iou_threshold": opts.tracker_iou,
                                "max_id": 0,
                            },
                        }
                    }
                ],
            }
        },
    ]
    operators: Dict[str, Any] = {
        "decodeyolopose": {"class": "DecodeYoloPose", "class_path": f"{_FW}/ax_models/decoders/yolopose.py"},
    }

    if spec.face:
        pipeline.append(
            {
                TASK_FACES: {
                    "model_name": _MODEL_DEFS["retinaface"]["name"],
                    "input": {
                        "type": "image",
                        "source": "no_tiles",
                        "color_format": "BGR",
                        "imreader_backend": "OpenCV",
                    },
                    "preprocess": [
                        {"letterbox": {"width": 640, "height": 640, "scaleup": True}},
                        {"torch-totensor": {"scale": False}},
                        {"linear-scaling": {"mean": "1, 1, 1", "shift": "-104, -117, -123"}},
                    ],
                    "inference": {"handle_all": False},
                    "postprocess": [
                        {
                            "decode-retinaface": {
                                "box_format": "xywh",
                                "normalized_coord": True,
                                "max_nms_boxes": 5000,
                                "conf_threshold": opts.face_conf,
                                "nms_iou_threshold": 0.4,
                                "nms_class_agnostic": True,
                                "nms_top_k": 200,
                            }
                        }
                    ],
                }
            }
        )
        pipeline.append(
            {
                TASK_FACE_EMB: {
                    "model_name": _MODEL_DEFS["facenet"]["name"],
                    "input": {
                        "type": "image",
                        "source": "roi",
                        "where": TASK_FACES,
                        "which": "AREA",
                        "top_k": opts.max_faces_per_frame,
                        "min_width": opts.min_face_px,
                        "min_height": opts.min_face_px,
                        "image_processing_on_roi": [
                            {"facealign": {"width": 192, "height": 192, "use_self_normalizing": False}}
                        ],
                    },
                    "preprocess": [
                        {"resize": {"width": 192, "height": 192}},
                        {"centercrop": {"width": 160, "height": 160}},
                        {"torch-totensor": None},
                        {"normalize": {"mean": "0.5, 0.5, 0.5", "std": "0.501960784, 0.501960784, 0.501960784"}},
                    ],
                    "postprocess": [{"decodeembeddings": None}],
                }
            }
        )
        operators["decode-retinaface"] = {
            "class": "DecodeRetinaface",
            "class_path": f"{_FW}/ax_models/decoders/retinaface.py",
        }

    if spec.lpr:
        pipeline.append(
            {
                TASK_PLATES: {
                    "model_name": _MODEL_DEFS["plate_detector"]["name"],
                    "input": {"type": "image", "source": "no_tiles"},
                    "preprocess": [
                        {"letterbox": {"width": 640, "height": 640, "scaleup": True}},
                        {"torch-totensor": None},
                    ],
                    "inference": {"handle_all": False},
                    "postprocess": [
                        {
                            "decodeyolo": {
                                "box_format": "xywh",
                                "normalized_coord": False,
                                "max_nms_boxes": 30000,
                                "conf_threshold": opts.plate_conf,
                                "nms_iou_threshold": 0.45,
                                "nms_class_agnostic": True,
                                "nms_top_k": 300,
                                "use_multi_label": False,
                            }
                        }
                    ],
                }
            }
        )
        pipeline.append(
            {
                TASK_PLATE_TEXT: {
                    "model_name": _MODEL_DEFS["lprnet"]["name"],
                    "input": {
                        "type": "image",
                        "source": "roi",
                        "where": TASK_PLATES,
                        "which": "AREA",
                        "top_k": opts.max_plates_per_frame,
                        "color_format": "RGB",
                    },
                    "preprocess": [
                        {"resize": {"width": 94, "height": 24}},
                        {"torch-totensor": {"scale": False}},
                        {"normalize": {"mean": 127.5, "std": 128}},
                    ],
                    "postprocess": [{"ctc-decoder": None}],
                }
            }
        )
        operators["decodeyolo"] = {"class": "DecodeYolo", "class_path": f"{_FW}/ax_models/decoders/yolo.py"}

    models: Dict[str, Any] = {}
    datasets: Dict[str, Any] = {}
    for key in model_keys:
        entry = _MODEL_DEFS[key]
        mdef = copy.deepcopy(entry["def"])
        extra = mdef.setdefault("extra_kwargs", {})
        extra["aipu_cores"] = cores[key]
        extra["max_compiler_cores"] = cores[key]
        for k, v in opts.model_overrides.get(key, {}).items():
            if v is None:
                mdef.pop(k, None)
            else:
                mdef[k] = v
        models[entry["name"]] = mdef
        ds_name = mdef["dataset"]
        ds = copy.deepcopy(_DATASETS[ds_name])
        if opts.calibration_dir and ds_name in _CAMERA_CALIBRATED_DATASETS:
            ds["repr_imgs_dir_path"] = str(Path(opts.calibration_dir).resolve())
        datasets[ds_name] = ds
    models["tracker"] = {"model_type": "CLASSICAL_CV", "task_category": "ObjectTracking"}

    return {
        "axelera-model-format": "1.0.0",
        "name": spec.name,
        "description": (
            f"Upside person/pose ({spec.pose_model}) + OC-SORT"
            + (" + RetinaFace/FaceNet face embeddings" if spec.face else "")
            + (" + YOLOv8n plate/LPRNet OCR" if spec.lpr else "")
            + ", INT8 PTQ on Metis AIPU"
        ),
        "pipeline": pipeline,
        "operators": operators,
        "models": models,
        "datasets": datasets,
    }


class _Dumper(yaml.SafeDumper):
    pass


def _none_representer(dumper, _):
    # Voyager uses bare keys (e.g. "- torch-totensor:") for operators without options.
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


_Dumper.add_representer(type(None), _none_representer)


def write_network(spec: NetworkSpec, out_dir: str | Path, opts: Optional[BuildOptions] = None) -> Path:
    """Writes the network YAML to ``out_dir/<name>.yaml`` and returns the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{spec.name}.yaml"
    with open(path, "w") as f:
        yaml.dump(build_network(spec, opts), f, Dumper=_Dumper, sort_keys=False, default_flow_style=False)
    return path


def all_specs() -> List[NetworkSpec]:
    return [NetworkSpec(p, face, lpr) for p in POSE_MODELS for face in (False, True) for lpr in (False, True)]


def face_enroll_spec_yaml(opts: Optional[BuildOptions] = None) -> Dict[str, Any]:
    """Standalone RetinaFace -> FaceNet network used to enroll identities from still images."""
    full = build_network(NetworkSpec("yolov8npose", face=True), opts)
    pipeline = [t for t in full["pipeline"] if next(iter(t)) in (TASK_FACES, TASK_FACE_EMB)]
    # Enrollment images are single faces: no tiling marker needed, 1 face is enough.
    pipeline[0][TASK_FACES]["input"].pop("source", None)
    pipeline[1][TASK_FACE_EMB]["input"]["top_k"] = 1
    models = {k: v for k, v in full["models"].items() if k in (_MODEL_DEFS["retinaface"]["name"], _MODEL_DEFS["facenet"]["name"])}
    models[_MODEL_DEFS["retinaface"]["name"]]["extra_kwargs"].update({"aipu_cores": 1, "max_compiler_cores": 1})
    models[_MODEL_DEFS["facenet"]["name"]]["extra_kwargs"].update({"aipu_cores": 1, "max_compiler_cores": 1})
    return {
        "axelera-model-format": "1.0.0",
        "name": f"{NETWORK_PREFIX}-face-enroll",
        "description": "RetinaFace -> FaceNet embedding network for identity enrollment",
        "pipeline": pipeline,
        "operators": {"decode-retinaface": full["operators"]["decode-retinaface"]},
        "models": models,
        "datasets": {k: v for k, v in full["datasets"].items() if k in ("WiderFace", "LFWTorchvisionPeople")},
    }


def write_face_enroll_network(out_dir: str | Path, opts: Optional[BuildOptions] = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NETWORK_PREFIX}-face-enroll.yaml"
    with open(path, "w") as f:
        yaml.dump(face_enroll_spec_yaml(opts), f, Dumper=_Dumper, sort_keys=False, default_flow_style=False)
    return path
