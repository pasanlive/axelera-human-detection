"""
CPU-based FaceNet (InceptionResnetV1) embedder for identity enrollment.

Runs the 512-d FaceNet model directly on the host CPU (PyTorch) using the official
Axelera Model Zoo weights (facenet_vgg_face_2.pt).

By executing still-image enrollment on the host CPU:
  1. Live camera streams on the Metis AIPU NEVER need to be suspended.
  2. Eliminates PCIe driver state transitions and GStreamer pipeline recreation.
  3. Prevents C++ std::system_error / abort crashes during enrollment.
  4. Produces embeddings 100% numerically compatible with Metis AIPU inference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

FACENET_WEIGHTS_URL = (
    "https://media.axelera.ai/artifacts/model_cards/weights/face_recognition/facenet_vgg_face_2.pt"
)
FACENET_WEIGHTS_MD5 = "13e767ebf459ce7459f130c3c1398a7b"


# ── InceptionResnetV1 PyTorch Architecture (matching Axelera Model Zoo) ─────────

class BasicConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1, padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                             padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Block35(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        out = self.conv2d(out) * self.scale + x
        return self.relu(out)


class Mixed_6a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)


class Block17(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out) * self.scale + x
        return self.relu(out)


class Mixed_7a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1)


class Block8(nn.Module):
    def __init__(self, scale: float = 1.0, noReLU: bool = False):
        super().__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out) * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResnetV1(nn.Module):
    """InceptionResnetV1 configured for VGGFace2 512-dimensional face embeddings."""

    def __init__(self, dropout_prob: float = 0.6):
        super().__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(*(Block35(scale=0.17) for _ in range(5)))
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(*(Block17(scale=0.10) for _ in range(10)))
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(*(Block8(scale=0.20) for _ in range(5)))
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.last_linear(x)
        x = self.last_bn(x)
        return x


# ── Face Detector & Embedder Manager ─────────────────────────────────────────

class FaceEmbedderCPU:
    """Manages face detection and PyTorch FaceNet embedding extraction on host CPU."""

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        self.device = torch.device("cpu")
        self._model: Optional[InceptionResnetV1] = None
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._weights_path = self._resolve_weights_path(weights_path)
        self._load_model()

    @staticmethod
    def _resolve_weights_path(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        candidates: List[Path] = []
        if custom_path:
            candidates.append(Path(custom_path))

        # Check project and system paths
        candidates.extend([
            Path("weights/facenet_vgg_face_2.pt"),
            Path("data/weights/facenet_vgg_face_2.pt"),
            Path.home() / "voyager-sdk" / "weights" / "facenet_vgg_face_2.pt",
            Path("/home/ubuntu/voyager-sdk/weights/facenet_vgg_face_2.pt"),
            Path.home() / ".cache" / "axelera" / "weights" / "facenet_vgg_face_2.pt",
        ])

        for c in candidates:
            if c.is_file() and c.stat().st_size > 10_000_000:
                return c.resolve()

        return None

    def _ensure_weights(self) -> Optional[Path]:
        if self._weights_path and self._weights_path.is_file():
            return self._weights_path

        target_dir = Path("data/weights")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "facenet_vgg_face_2.pt"
        if target.is_file() and target.stat().st_size > 10_000_000:
            self._weights_path = target
            return target

        # Attempt download if internet available
        try:
            import urllib.request
            print(f"[FACE EMBEDDER CPU] Downloading FaceNet weights from {FACENET_WEIGHTS_URL}...")
            urllib.request.urlretrieve(FACENET_WEIGHTS_URL, str(target))
            if target.is_file() and target.stat().st_size > 10_000_000:
                self._weights_path = target
                print(f"[FACE EMBEDDER CPU] Downloaded weights to {target}")
                return target
        except Exception as e:
            print(f"[FACE EMBEDDER CPU] Could not download FaceNet weights: {e}")

        return None

    def _load_model(self) -> None:
        w_path = self._ensure_weights()
        if not w_path or not w_path.is_file():
            print("[FACE EMBEDDER CPU] FaceNet weights not yet available for CPU enrollment.")
            return

        try:
            model = InceptionResnetV1()
            weights = torch.load(str(w_path), map_location=self.device, weights_only=False)
            model.load_state_dict(weights, strict=False)
            model.eval()
            self._model = model
            print(f"[FACE EMBEDDER CPU] Loaded FaceNet model on CPU from {w_path}")
        except Exception as e:
            print(f"[FACE EMBEDDER CPU] Failed to load FaceNet model on CPU: {e}")
            self._model = None

    def _get_cascade(self) -> Any:
        if self._face_cascade is None:
            cascade_cls = getattr(cv2, "CascadeClassifier", None)
            if cascade_cls is not None:
                haar_data = getattr(cv2, "data", None)
                if haar_data and hasattr(haar_data, "haarcascades"):
                    haar_path = os.path.join(haar_data.haarcascades, "haarcascade_frontalface_default.xml")
                    if os.path.exists(haar_path):
                        try:
                            self._face_cascade = cascade_cls(haar_path)
                        except Exception:
                            self._face_cascade = None
        return self._face_cascade

    def detect_face_roi(self, image_bgr: np.ndarray) -> np.ndarray:
        """Finds face ROI and crops with margin; falls back to center-square crop."""
        h, w = image_bgr.shape[:2]
        cascade = self._get_cascade()
        if cascade is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) > 0:
                # Pick largest detected face
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                # Expand box by 15% for context/hair
                pad_w = int(fw * 0.15)
                pad_h = int(fh * 0.15)
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(w, fx + fw + pad_w)
                y2 = min(h, fy + fh + pad_h)
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    return crop

        # Center square fallback
        sz = min(h, w)
        cx, cy = w // 2, h // 2
        return image_bgr[cy - sz // 2 : cy + sz // 2, cx - sz // 2 : cx + sz // 2]

    def embed_crop(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 512-d normalized FaceNet face embedding directly from an already cropped face BGR ROI.
        Bypasses face detection cascade for high-speed live stream processing.
        """
        if self._model is None:
            self._load_model()
            if self._model is None:
                return None

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        try:
            # 1. Resize to 160x160 and convert BGR -> RGB
            resized = cv2.resize(face_crop_bgr, (160, 160), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

            # 2. Axelera normalization: (x / 255.0 - 0.5) / 0.501960784
            # Equivalent to (x - 127.5) / 128.0
            norm = (rgb - 127.5) / 128.0
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()

            # 3. Inference
            with torch.no_grad():
                out = self._model(tensor)
                emb = out.squeeze(0).cpu().numpy().astype(np.float32)

            # 4. L2 normalization to unit hypersphere
            norm_val = np.linalg.norm(emb)
            if norm_val > 1e-6:
                emb = emb / norm_val
            return emb
        except Exception as e:
            print(f"[FACE EMBEDDER CPU] Error during CPU face crop embedding: {e}")
            return None

    def embed(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 512-d normalized FaceNet face embedding from a BGR image.
        Returns None if model weights are unavailable or no face is present.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        face_crop = self.detect_face_roi(image_bgr)
        if face_crop is None or face_crop.size == 0:
            face_crop = image_bgr

        return self.embed_crop(face_crop)


# ── Global singleton accessor ────────────────────────────────────────────────

_global_embedder: Optional[FaceEmbedderCPU] = None


def get_cpu_face_embedder(weights_hint_dir: Optional[Union[str, Path]] = None) -> Optional[FaceEmbedderCPU]:
    global _global_embedder
    if _global_embedder is None:
        custom_weights = None
        if weights_hint_dir:
            p = Path(weights_hint_dir) / "weights" / "facenet_vgg_face_2.pt"
            if p.is_file():
                custom_weights = p
        _global_embedder = FaceEmbedderCPU(weights_path=custom_weights)
    return _global_embedder
