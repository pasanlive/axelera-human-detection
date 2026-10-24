#!/usr/bin/env python3
"""
Measure real Metis throughput to calibrate the capacity planner.

Runs each compiled network over N copies of a local video file (no RTSP/decode
limits, unlimited frame rate) and reports aggregate FPS. From the pose-only
results it suggests ``capacity.derating`` and per-model ``capacity.model_fps``
values for config.yaml.

    source ~/voyager-sdk/venv/bin/activate
    python tools/benchmark_capacity.py --video data/sample_1080p.mp4 --streams 1 4 8 --seconds 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.voyager.capacity_planner import DEFAULT_MODEL_FPS  # noqa: E402
from src.voyager.network_builder import NetworkSpec, write_network  # noqa: E402
from src.voyager.sdk import sdk_modules  # noqa: E402
from src.voyager.stream_runtime import build_options_from_config  # noqa: E402


def run(sdk, network_yaml: str, sources, seconds: float, build_root=None) -> float:
    kwargs = dict(network=network_yaml, sources=sources, pipe_type="gst", aipu_cores=4,
                  allow_hardware_codec=True, specified_frame_rate=0, timeout=0)
    if build_root:
        kwargs["build_root"] = build_root
    stream = sdk.create_inference_stream(**kwargs)
    n, t0, warm = 0, None, 30
    try:
        for _ in stream:
            n += 1
            if n == warm:
                t0 = time.time()
            if t0 and time.time() - t0 >= seconds:
                break
    finally:
        stream.stop()
    return (n - warm) / (time.time() - t0) if t0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    ap.add_argument("--video", required=True, help="local video file (ideally from your cameras)")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--networks", nargs="+",
                    default=["upside-yolov8npose", "upside-yolov8spose", "upside-yolov8npose-face"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    vcfg = cfg.get("voyager", {}) or {}
    sdk = sdk_modules()
    out_dir = ROOT / vcfg.get("networks_dir", "voyager/networks")
    opts = build_options_from_config(vcfg)
    video = str(Path(args.video).resolve())

    results = {}
    for name in args.networks:
        y = write_network(NetworkSpec.from_name(name), out_dir, opts)
        for n in args.streams:
            fps = run(sdk, str(y), [video] * n, args.seconds, vcfg.get("build_root"))
            results[f"{name}x{n}"] = round(fps, 1)
            print(f"[bench] {name:32s} streams={n:2d}  aggregate={fps:7.1f} FPS")

    suggestions = {}
    for model in ("yolov8npose", "yolov8spose"):
        best = max((v for k, v in results.items() if k.startswith(f"upside-{model}x")), default=0)
        if best:
            suggestions[model] = round(best / DEFAULT_MODEL_FPS[model], 2)
    if suggestions:
        derating = round(min(suggestions.values()), 2)
        print(f"\nSuggested capacity.derating: {derating}  (measured/zoo per pose model: {suggestions})")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
