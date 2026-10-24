#!/usr/bin/env python3
"""
Capture representative frames from the configured cameras for INT8 calibration.

Post-training quantization accuracy depends on calibration data that looks like
production data (lighting, camera height, lens). This grabs frames spread over
time from every enabled camera (main stream) into ``voyager.calibration_dir``.

    python tools/collect_calibration.py --count 400 --interval 2

This is an offline utility (OpenCV capture on the host CPU); it does not run inference.
For best results run it at different times of day and append (--append).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    ap.add_argument("--count", type=int, default=400, help="total frames across all cameras")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between grabs per camera")
    ap.add_argument("--append", action="store_true", help="keep existing images")
    ap.add_argument("--sub", action="store_true", help="use sub streams instead of main")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    out = (ROOT / cfg.get("voyager", {}).get("calibration_dir", "data/calibration")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.append:
        for p in out.glob("*.jpg"):
            p.unlink()

    cams = [c for c in cfg.get("cameras", []) if c.get("enabled", True)]
    if not cams:
        sys.exit("No enabled cameras in config")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    per_cam = math.ceil(args.count / len(cams))
    caps = {}
    for c in cams:
        url = c.get("source_sub") if args.sub and c.get("source_sub") else c["source"]
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[calib] WARNING: cannot open {c['id']} ({url})")
            continue
        caps[c["id"]] = cap
    if not caps:
        sys.exit("No camera could be opened")

    saved = {cid: 0 for cid in caps}
    start_idx = len(list(out.glob("*.jpg")))
    next_due = {cid: 0.0 for cid in caps}
    print(f"[calib] Capturing {per_cam} frames/camera from {len(caps)} camera(s) into {out}")
    while any(saved[c] < per_cam for c in caps):
        now = time.time()
        for cid, cap in caps.items():
            ok = cap.grab()  # keep the buffer drained
            if not ok or saved[cid] >= per_cam or now < next_due[cid]:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            next_due[cid] = now + args.interval
            cv2.imwrite(str(out / f"{cid}_{start_idx + saved[cid]:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved[cid] += 1
        sys.stdout.write("\r[calib] " + "  ".join(f"{c}:{n}/{per_cam}" for c, n in saved.items()))
        sys.stdout.flush()
    for cap in caps.values():
        cap.release()
    print(f"\n[calib] Done. Now run: python tools/deploy_networks.py")


if __name__ == "__main__":
    main()
