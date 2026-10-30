"""
Converts Voyager ``FrameResult`` metadata into plain, numpy-backed records.

All decoding/NMS already happened in Voyager's C++ decoders on the pipeline
threads; this module only does cheap, vectorised glue:

* associate YOLOv8-pose detections with OC-SORT tracks (IoU matrix)
* associate RetinaFace faces with person tracks (upper-body containment)
* extract FaceNet embeddings / LPRNet plate strings from secondary metas

It deliberately duck-types the SDK meta classes so it can be unit-tested with
lightweight fakes and used by the SimulatedRuntime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.voyager.network_builder import (
    TASK_FACE_EMB,
    TASK_FACES,
    TASK_PERSONS,
    TASK_PLATE_TEXT,
    TASK_PLATES,
    TASK_TRACKS,
)


@dataclass
class FrameAnalysis:
    """Inference output for one frame of one camera."""

    stream_id: int
    timestamp: float
    frame_size: Tuple[int, int] = (0, 0)  # (w, h)
    persons: List[Dict[str, Any]] = field(default_factory=list)
    """[{bbox: [x1,y1,x2,y2], confidence, keypoints: (17,3) ndarray | None, track_id: int | None}]"""
    faces: List[Dict[str, Any]] = field(default_factory=list)
    """[{bbox, confidence, embedding: (D,) float32 | None, person_index: int | None}]"""
    plates: List[Dict[str, Any]] = field(default_factory=list)
    """[{bbox, confidence, plate_text: str}]"""


# ── vectorised geometry ──────────────────────────────────────────────────────

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def greedy_match(scores: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Greedy one-to-one assignment on a score matrix (higher is better)."""
    pairs: List[Tuple[int, int]] = []
    if scores.size == 0:
        return pairs
    s = scores.copy()
    while True:
        idx = int(np.argmax(s))
        r, c = divmod(idx, s.shape[1])
        if s[r, c] < threshold:
            break
        pairs.append((r, c))
        s[r, :] = -1.0
        s[:, c] = -1.0
    return pairs


def assign_faces_to_persons(face_boxes: np.ndarray, person_boxes: np.ndarray,
                            head_fraction: float = 0.45) -> np.ndarray:
    """For each face, index of the smallest person whose upper body contains the face centre (-1 if none)."""
    f = np.asarray(face_boxes, dtype=np.float32).reshape(-1, 4)
    p = np.asarray(person_boxes, dtype=np.float32).reshape(-1, 4)
    out = np.full(f.shape[0], -1, dtype=np.int64)
    if f.size == 0 or p.size == 0:
        return out
    cx = (f[:, 0] + f[:, 2]) * 0.5
    cy = (f[:, 1] + f[:, 3]) * 0.5
    head_bottom = p[:, 1] + (p[:, 3] - p[:, 1]) * head_fraction
    inside = (
        (cx[:, None] >= p[None, :, 0]) & (cx[:, None] <= p[None, :, 2])
        & (cy[:, None] >= p[None, :, 1]) & (cy[:, None] <= head_bottom[None, :])
    )
    area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    cost = np.where(inside, area[None, :], np.inf)
    best = np.argmin(cost, axis=1)
    has = np.isfinite(cost[np.arange(f.shape[0]), best])
    out[has] = best[has]
    return out


# ── meta extraction helpers ──────────────────────────────────────────────────

def _meta_get(meta: Any, task: str) -> Any:
    if meta is None:
        return None
    try:
        return meta.get(task) if hasattr(meta, "get") else meta[task]
    except (KeyError, TypeError):
        return None


def _arr(x: Any, cols: Optional[int] = None) -> np.ndarray:
    a = np.asarray(x if x is not None else [], dtype=np.float32)
    if cols is not None:
        a = a.reshape(-1, cols)
    return a


def _secondary(task_meta: Any, index: int, secondary_task: str) -> Any:
    """Secondary meta for object ``index`` (FaceNet embedding / LPRNet text)."""
    try:
        objs = task_meta.objects
        return objs[index].get_secondary_meta(secondary_task)
    except Exception:
        pass
    try:
        return task_meta.get_secondary_meta(secondary_task, index)
    except Exception:
        return None


def _track_boxes(tracks: Any) -> Tuple[np.ndarray, np.ndarray]:
    """(track_ids (M,), boxes (M,4)) for tracks present in this frame."""
    if tracks is None:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    boxes = getattr(tracks, "boxes", None)
    if isinstance(boxes, dict) and boxes:
        ids = np.fromiter(boxes.keys(), dtype=np.int64)
        return ids, _arr(list(boxes.values()), 4)
    hist = getattr(tracks, "tracking_history", None) or {}
    ids, bxs = [], []
    for obj in getattr(tracks, "objects", []) or []:
        tid = int(obj.track_id)
        h = hist.get(tid)
        if h is not None and len(h):
            ids.append(tid)
            bxs.append(np.asarray(h[-1], dtype=np.float32)[:4])
    return np.asarray(ids, dtype=np.int64), _arr(bxs, 4)


def parse_frame_result(result: Any, timestamp: float, track_iou: float = 0.3) -> FrameAnalysis:
    """FrameResult -> FrameAnalysis."""
    meta = getattr(result, "meta", None)
    img = getattr(result, "image", None)
    size = (0, 0)
    if img is not None:
        for getter in (lambda i: (i.width, i.height), lambda i: tuple(i.size)[:2],
                       lambda i: tuple(i.shape[:2])[::-1]):
            try:
                w, h = getter(img)
                size = (int(w), int(h))
                break
            except Exception:
                continue
    fa = FrameAnalysis(stream_id=int(getattr(result, "stream_id", 0)), timestamp=timestamp, frame_size=size)

    # 1. Native tracker support (e.g. from yolov5m-v7-coco-tracker)
    tracker = getattr(result, "pedestrian_and_vehicle_tracker", None) or _meta_get(meta, "pedestrian_and_vehicle_tracker")
    if tracker is not None:
        try:
            for veh in tracker:
                class_id = getattr(veh, "class_id", -1)
                # Class 0 in COCO dataset is 'person' / human
                if class_id == 0 or getattr(veh, "label", "").lower() == "person":
                    history = getattr(veh, "history", None)
                    if history is not None and len(history) > 0:
                        bbox = history[-1].tolist() if hasattr(history[-1], "tolist") else list(history[-1])
                    else:
                        bbox = getattr(veh, "bbox", [0, 0, 0, 0])
                    score = float(getattr(veh, "score", getattr(veh, "confidence", 1.0)))
                    track_id = int(getattr(veh, "track_id", -1))
                    fa.persons.append({
                        "bbox": list(bbox),
                        "confidence": score,
                        "keypoints": None,
                        "track_id": track_id if track_id >= 0 else None,
                    })
        except Exception:
            pass

    # 2. Raw detections fallback (e.g. from yolov5m detections)
    if not fa.persons:
        dets = getattr(result, "detections", None) or _meta_get(meta, "detections")
        if dets is not None:
            try:
                for det in dets:
                    class_id = getattr(det, "class_id", -1)
                    label = getattr(det, "label", "")
                    if class_id == 0 or label.lower() == "person":
                        bbox = getattr(det, "bbox", [0, 0, 0, 0])
                        score = float(getattr(det, "score", getattr(det, "confidence", 1.0)))
                        fa.persons.append({
                            "bbox": list(bbox) if hasattr(bbox, "tolist") else list(bbox),
                            "confidence": score,
                            "keypoints": None,
                            "track_id": None,
                        })
            except Exception:
                pass

    # 3. Persons + keypoints from custom pipeline (TASK_PERSONS)
    if not fa.persons:
        persons = _meta_get(meta, TASK_PERSONS)
        p_boxes = _arr(getattr(persons, "boxes", None), 4) if persons is not None else np.zeros((0, 4), np.float32)
        p_scores = _arr(getattr(persons, "scores", None)) if persons is not None else np.zeros(0, np.float32)
        kpts = getattr(persons, "keypoints", None) if persons is not None else None
        kpts = np.asarray(kpts, dtype=np.float32) if kpts is not None and len(kpts) else None
        if kpts is not None:
            kpts = kpts.reshape(len(p_boxes), -1, 3) if kpts.size == len(p_boxes) * 51 else kpts

        # Track association
        tids, t_boxes = _track_boxes(_meta_get(meta, TASK_TRACKS))
        track_of = np.full(len(p_boxes), -1, dtype=np.int64)
        for r, c in greedy_match(iou_matrix(p_boxes, t_boxes), track_iou):
            track_of[r] = tids[c]

        for i in range(len(p_boxes)):
            fa.persons.append({
                "bbox": p_boxes[i].tolist(),
                "confidence": float(p_scores[i]) if i < len(p_scores) else 0.0,
                "keypoints": kpts[i] if kpts is not None and i < len(kpts) else None,
                "track_id": int(track_of[i]) if track_of[i] >= 0 else None,
            })

    # Faces + embeddings
    faces = _meta_get(meta, TASK_FACES)
    if faces is not None:
        f_boxes = _arr(getattr(faces, "boxes", None), 4)
        f_scores = _arr(getattr(faces, "scores", None))
        owner = assign_faces_to_persons(f_boxes, p_boxes)
        for i in range(len(f_boxes)):
            emb = None
            em = _secondary(faces, i, TASK_FACE_EMB)
            vecs = getattr(em, "embedding", None) if em is not None else None
            if vecs is not None and len(vecs):
                emb = np.asarray(vecs[0], dtype=np.float32).ravel()
            fa.faces.append({
                "bbox": f_boxes[i].tolist(),
                "confidence": float(f_scores[i]) if i < len(f_scores) else 0.0,
                "embedding": emb,
                "person_index": int(owner[i]) if owner[i] >= 0 else None,
            })

    # Plates + OCR
    plates = _meta_get(meta, TASK_PLATES)
    if plates is not None:
        pl_boxes = _arr(getattr(plates, "boxes", None), 4)
        pl_scores = _arr(getattr(plates, "scores", None))
        for i in range(len(pl_boxes)):
            tm = _secondary(plates, i, TASK_PLATE_TEXT)
            text = getattr(tm, "label", None) if tm is not None else None
            fa.plates.append({
                "bbox": pl_boxes[i].tolist(),
                "confidence": float(pl_scores[i]) if i < len(pl_scores) else 0.0,
                "plate_text": (text or "").strip(),
            })
    return fa
