# Multi-Camera Human Detection, Pose Estimation & Face Recognition for Axelera Metis

Multi-camera (1–15 RTSP) person analytics on an **Axelera Metis AIPU** with a **Rockchip RK3588** host, built on **Voyager SDK 1.6.0**.

- 🚶 **Person detection + 17-keypoint pose** (YOLOv8n/s-pose, INT8)
- 👤 **Face recognition** (RetinaFace → FaceNet VGGFace2 embeddings → local gallery)
- 🚗 **License plate recognition** (plate detector → LPRNet, optional)
- 🎯 **Tracking** (Voyager tracker, per camera) + zone line/polygon crossing
- 📈 **Dynamic capacity planning**: high resolution and FPS with few cameras, scaling down automatically as cameras or models are added
- 🖥️ **Web dashboard** with MJPEG preview, model toggles, capacity profile and camera management

> [!IMPORTANT]
> All inference runs on the Metis AIPU as INT8 models compiled by the Voyager `deploy` tool. **There is no CPU inference fallback.** Without the SDK and compiled networks the app fails at startup. Use `--simulate` for UI development.

---

## 🛠️ Architecture (Voyager-owned pipeline)

```
 RTSP main/sub ──► GStreamer (rtspsrc → decodebin, MPP HW decode on RK3588 VPU)
                     │  zero-copy resize / letterbox / colour convert
                     ▼
        ┌────────────────────────────────────────────────────┐
        │ Metis AIPU (4 cores, INT8)  — compiled .axnet/.axm │
        │  YOLOv8-pose ─► tracker ─► RetinaFace ─► FaceNet   │
        │                        └─► plate det ─► LPRNet     │
        └──────────────────────────┬─────────────────────────┘
                                   │ FrameResult (AxMeta + image)
                                   ▼
   src/voyager/stream_runtime.py ──► result_parser ──► pipeline.py
     (create_inference_stream,         (tracks, faces,    (identity, zones,
      add/remove source & pipeline)     plates, embeds)    plates, events)
                                   │
                                   ▼
             src/render/preview.py (renders only when a viewer is connected)
                                   ▼
                       src/web/server.py (Flask / native HTTP)
```

| Module | Role |
|---|---|
| [network_builder.py](src/voyager/network_builder.py) | Generates the Voyager network YAMLs (`upside-*`) for each feature combination |
| [capacity_planner.py](src/voyager/capacity_planner.py) | Picks the tier, FPS, main/sub stream and tiling from the AIPU, VPU-decode and host budgets |
| [stream_runtime.py](src/voyager/stream_runtime.py) | Owns the `InferenceStream`; adds/removes cameras incrementally; rebuilds on tier change |
| [result_parser.py](src/voyager/result_parser.py) | Converts `AxMeta` into tracks, faces, plates and embeddings |
| [pipeline.py](src/pipeline.py) | Public API: models, cameras, capacity, enrollment, stats; debounced re-planning |

### Capacity tiers

The planner picks the best tier that fits under `capacity.target_utilization`:

| Tier | Pose model | Stream | Notes |
|---|---|---|---|
| `ultra` | yolov8s-pose | main | Tiled for small and distant people when budget allows |
| `high` | yolov8s-pose | main | |
| `standard` | yolov8n-pose | main | |
| `balanced` | yolov8n-pose | main/sub | |
| `economy` | yolov8n-pose | sub | |
| `minimal` | yolov8n-pose | sub | Floor of 3 FPS/camera |

These are rough planner estimates. Calibrate them on the board with `tools/benchmark_capacity.py`:

| Load | Plan |
|---|---|
| 1 camera, pose only | `ultra` @ 25 FPS |
| 15 cameras, pose only | `high` @ 16 FPS |
| 10 cameras, + face | `balanced` @ 9 FPS |
| 15 cameras, + face | `economy` @ 6 FPS |
| 15 cameras, + face + LPR | `minimal` @ 4 FPS |

Set `profile: auto` for dynamic planning, or pin a tier from the dashboard or `POST /api/capacity/profile`.

---

## 📋 Requirements

- Metis M.2/PCIe card on an RK3588 host (Ubuntu 22.04 image)
- **Voyager SDK 1.6.0** installed and activated (`source <voyager-sdk>/venv/bin/activate`), with `AXELERA_FRAMEWORK` set
- Rockchip GStreamer plugins for hardware decode: `gstreamer1.0-rockchip` (`mppvideodec`). Without them decode falls back to software and the CPU becomes the bottleneck.
- Python deps: `pip install -r requirements.txt` (`requirements-dev.txt` for tests)

```bash
gst-inspect-1.0 mppvideodec   # must succeed for zero-copy HW decode
```

---

## 🚀 Workflow

### 1. Configure cameras
In [config/config.yaml](config/config.yaml), each camera has a main and a sub RTSP stream (Hikvision format shown):

```yaml
cameras:
- id: cam_01
  name: Front Entrance
  enabled: true
  source: rtsp://user:pass@192.168.0.102:554/h264/ch1/main/av_stream
  source_sub: rtsp://user:pass@192.168.0.102:554/h264/ch1/sub/av_stream
  main_resolution: [1920, 1080]
  sub_resolution: [640, 360]
  native_fps: 25
```

Other sections:
- `voyager:` holds the SDK path, build root, HW decode, RTSP transport and latency, thresholds and tracker settings.
- `capacity:` holds the profile, budgets and per-model FPS.
- `preview:` holds the MJPEG FPS, width and quality.
- `zone_crossing.zones` must set `camera_id`. Points are in **main-stream pixels** and are scaled automatically when the planner uses the sub stream.

> [!NOTE]
> The dashboard and auto-updater rewrite `config.yaml` with `yaml.safe_dump`, which removes comments. Keep notes here rather than in the YAML.

### 2. Collect calibration images (INT8 PTQ)
```bash
python tools/collect_calibration.py --count 400 --interval 2   # frames from your own cameras
```
Representative frames from your cameras give much better INT8 accuracy than generic datasets.

### 3. Quantize and compile the networks
```bash
python tools/deploy_networks.py --dry-run        # print the plan and YAMLs only
python tools/deploy_networks.py --all            # all feature combinations + face-enroll network
python tools/deploy_networks.py --network upside-yolov8npose-face
```
This writes the network YAMLs to `$AXELERA_FRAMEWORK/customers/upside/` (`voyager.networks_dir`, relative to the SDK). They must live inside the SDK checkout, because `deploy.py` only accepts networks it finds there. It then runs Voyager `deploy.py <network-name>` (PTQ → compile), which outputs to `$AXELERA_FRAMEWORK/build` (or `voyager.build_root`).

> [!TIP]
> Compiling is slow on the RK3588. You can compile on an x86 host that has the same SDK version, then copy the `build/<network>` directories to the board.

### 4. Run
```bash
python app.py                      # HTTPS dashboard on :8000
python app.py --no-https --port 8010
python app.py --print-plan         # print the capacity plan for the current config and exit (no SDK needed)
python app.py --simulate           # synthetic results, no hardware (UI development)
```

### 5. Face enrollment
```bash
python app.py --enroll "John Doe" --image path/to/john.jpg
python app.py --list-faces
```
You can also enroll from the dashboard, using an uploaded image or a live camera crop.

> [!WARNING]
> The face DB format changed to FaceNet (VGGFace2) embeddings. Old ArcFace galleries are backed up to `data/face_db.legacy.json`, and **every identity must be re-enrolled**.

### 6. Calibrate the capacity budgets
```bash
python tools/benchmark_capacity.py --video sample.mp4 --streams 1 4 8 --seconds 20
```
Copy the measured per-network FPS into `capacity.model_fps` so that auto-planning matches your board.

---

## 🌐 HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Cameras, telemetry (core temp, device FPS, estimated AIPU load), stats |
| GET | `/api/models` · POST `/api/models/toggle` | Model status / enable/disable (`{model_id, enabled, camera_id?}`) |
| GET | `/api/capacity` · POST `/api/capacity/profile` | Active plan / set `{profile}` (`auto` or a tier) |
| GET/POST | `/api/cameras` | List / add camera |
| DELETE | `/api/cameras/<id>` | Remove camera |
| POST | `/api/cameras/<id>/enable` | Enable/disable camera |
| GET | `/api/events` | Recent recognition/zone/plate events |
| GET | `/api/identities` · DELETE `/api/identity/<name>` · POST `/api/enroll_face` | Face gallery |
| GET | `/api/zones`, `/api/zone_events` · POST `/api/zones/config` | Zone crossing |
| GET | `/api/plates`, `/api/plates/unique` · DELETE `/api/plates/<plate>` | Plate log |
| GET | `/video_feed` (grid), `/video_feed/<cam_id>`, `/api/snapshot` | MJPEG preview / JPEG snapshot |
| GET/POST | `/api/update/*` | Auto-updater |

Model IDs: `human_detector`, `pose_estimator`, `face_recognizer`, `license_plate_recognizer`, `zone_crossing`. Enabling face or LPR changes the compiled network, so it triggers a re-plan.

---

## ⚠️ Known limitations / to validate on the board

- Tiling (`ultra`) combined with keypoint models has not been verified on 1.6.0. If it fails, the runtime disables tiling automatically and re-plans.
- Running several pipelines at once (mixed feature groups) on the 4 AIPU cores is not yet benchmarked.
- The LPRNet Model Zoo weights are trained on CCPD (Chinese plates). Retrain them for local plates.
- AIPU load is **estimated** (planned utilisation × achieved/planned FPS) because 1.6.0 has no per-core utilisation tracer.
- Enrolling from an uploaded image opens a short-lived second inference stream.
- The OpenCV `--gui` window mode was removed. Use the web dashboard.

---

## 🧪 Tests
```bash
python -m unittest discover tests     # runs without the SDK (simulated runtime)
```

## 📄 License
MIT License. Built for Axelera Metis AIPU & Voyager SDK.
