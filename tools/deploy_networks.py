#!/usr/bin/env python3
"""
Generate, quantize (INT8 PTQ) and compile the application's Voyager networks.

Run on the Metis host (or an x86 build host with the same Voyager SDK version)
inside the SDK virtualenv:

    source ~/voyager-sdk/venv/bin/activate
    python tools/collect_calibration.py            # optional but recommended: 400 frames from your cameras
    python tools/deploy_networks.py                # networks needed by the configured cameras
    python tools/deploy_networks.py --all          # every pose/face/lpr combination (8 + enrollment)
    python tools/deploy_networks.py --network upside-yolov8spose-face
    python tools/deploy_networks.py --dry-run      # only write the YAMLs

Compiling is CPU/RAM heavy. If it is too slow on the RK3588, compile on an x86
host and copy ``<build_root>/<network>/`` directories to the board.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.voyager.capacity_planner import CapacityBudgets, CapacityPlanner, CameraSpec  # noqa: E402
from src.voyager.network_builder import (  # noqa: E402
    NetworkSpec,
    all_specs,
    face_enroll_spec_yaml,
    write_face_enroll_network,
    write_network,
)
from src.voyager.sdk import framework_path  # noqa: E402
from src.voyager.stream_runtime import build_options_from_config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def needed_specs(cfg: dict) -> list:
    models = cfg.get("models", {}) or {}
    face = models.get("face_recognizer", {}).get("enabled", True)
    lpr = models.get("license_plate_recognizer", {}).get("enabled", False)
    cams = []
    for c in cfg.get("cameras", []) or []:
        ov = c.get("features") or {}
        feats = set()
        if ov.get("face_recognizer", face):
            feats.add("face")
        if ov.get("license_plate_recognizer", lpr):
            feats.add("lpr")
        cams.append(CameraSpec(c["id"], c.get("name", c["id"]), c["source"], features=frozenset(feats)))
    planner = CapacityPlanner(CapacityBudgets.from_config(cfg.get("capacity", {}) or {}))
    specs = set(planner.required_specs(cams))
    # Always have the plain pose networks so toggling face/LPR off never needs a compile.
    specs |= {NetworkSpec("yolov8npose"), NetworkSpec("yolov8spose")}
    return sorted(specs, key=lambda s: s.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    ap.add_argument("--all", action="store_true", help="deploy all 8 combinations")
    ap.add_argument("--network", action="append", default=[], help="deploy only this network (repeatable)")
    ap.add_argument("--no-enroll", action="store_true", help="skip the face-enrollment network")
    ap.add_argument("--num-cal-images", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="write YAMLs only")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    vcfg = cfg.get("voyager", {}) or {}
    os.chdir(ROOT)
    opts = build_options_from_config(vcfg)
    out_dir = (ROOT / vcfg.get("networks_dir", "voyager/networks")).resolve()
    n_cal = args.num_cal_images or int(vcfg.get("num_calibration_images", 400))

    cal_dir = vcfg.get("calibration_dir")
    n_imgs = sum(1 for p in Path(cal_dir).glob("*") if p.suffix.lower() in IMG_EXT) if cal_dir and Path(cal_dir).is_dir() else 0
    if n_imgs:
        print(f"[deploy] Calibrating pose/face/plate detectors on {n_imgs} camera frames from {cal_dir}")
        if n_imgs < 100:
            print("[deploy] WARNING: <100 calibration images; INT8 accuracy may suffer. Run tools/collect_calibration.py")
        n_cal = min(n_cal, n_imgs)
    else:
        print("[deploy] No camera calibration frames found - using Model Zoo representative images.")

    if args.network:
        specs = [NetworkSpec.from_name(n) for n in args.network if n != face_enroll_spec_yaml()["name"]]
        want_enroll = face_enroll_spec_yaml()["name"] in args.network
    else:
        specs = all_specs() if args.all else needed_specs(cfg)
        want_enroll = not args.no_enroll and (cfg.get("models", {}).get("face_recognizer", {}).get("enabled", True)
                                              or args.all)

    yamls = [write_network(s, out_dir, opts) for s in specs]
    if want_enroll:
        yamls.append(write_face_enroll_network(out_dir, opts))
    for y in yamls:
        print(f"[deploy] wrote {y.relative_to(ROOT)}")
    if args.dry_run:
        return

    fw = framework_path(vcfg.get("sdk_path"))
    if not fw:
        sys.exit("[deploy] Voyager SDK not found: set voyager.sdk_path or AXELERA_FRAMEWORK (dir containing deploy.py)")
    build_root = vcfg.get("build_root")
    failures = []
    for y in yamls:
        cmd = [sys.executable, os.path.join(fw, "deploy.py"), str(y), "--num-cal-images", str(n_cal)]
        if build_root:
            cmd += ["--build-root", str(Path(build_root).expanduser().resolve())]
        print(f"\n[deploy] $ {' '.join(cmd)}")
        env = dict(os.environ, AXELERA_FRAMEWORK=fw)
        if subprocess.run(cmd, cwd=fw, env=env).returncode != 0:
            failures.append(y.stem)
    if failures:
        sys.exit(f"[deploy] FAILED: {failures}")
    print("\n[deploy] All networks quantized (INT8 PTQ) and compiled for Metis.")


if __name__ == "__main__":
    main()
