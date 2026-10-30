"""
Unit tests for Axelera Metis Multi-Camera System components.

Runs without Metis hardware or the Voyager SDK: the pipeline uses SimulatedRuntime
and the result parser is exercised with lightweight fakes of the SDK meta classes.
"""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import yaml

from src.analytics.identity import TrackIdentityResolver
from src.pipeline import MultiCameraPipeline
from src.utils.face_db import FaceDatabase
from src.utils.visualization import Visualizer
from src.voyager.capacity_planner import (
    TIERS,
    CameraSpec,
    CapacityBudgets,
    CapacityPlanner,
    tile_count,
)
from src.voyager.network_builder import (
    TASK_FACE_EMB,
    TASK_FACES,
    TASK_PERSONS,
    TASK_PLATE_TEXT,
    TASK_PLATES,
    TASK_TRACKS,
    BuildOptions,
    NetworkSpec,
    all_specs,
    build_network,
    face_enroll_spec_yaml,
    write_network,
)
from src.voyager.result_parser import assign_faces_to_persons, iou_matrix, parse_frame_result
from src.web.server import WebServer

TIER_RANK = {t.name: i for i, t in enumerate(TIERS)}


def _cams(n, feats=frozenset(), sub=True):
    return [CameraSpec(f"cam_{i:02d}", f"Cam {i}", f"rtsp://cam{i}/main",
                       f"rtsp://cam{i}/sub" if sub else None, features=feats) for i in range(n)]


def _sim_config(tmpdir, n_cams=2, face=True, lpr=False):
    return {
        "cameras": [{"id": f"cam_{i:02d}", "name": f"Cam {i}", "enabled": True,
                     "source": f"rtsp://cam{i}/main", "source_sub": f"rtsp://cam{i}/sub"} for i in range(n_cams)],
        "models": {"human_detector": {"enabled": True}, "pose_estimator": {"enabled": True},
                   "face_recognizer": {"enabled": face, "match_threshold": 0.5},
                   "license_plate_recognizer": {"enabled": lpr}},
        "face_db": {"path": os.path.join(tmpdir, "face_db.json")},
        "plate_db": {"path": os.path.join(tmpdir, "plate_db.json")},
        "capacity": {"profile": "auto"},
        "preview": {"fps": 30, "grid_fps": 30},
    }


class TestNetworkBuilder(unittest.TestCase):

    def test_all_specs_valid(self):
        specs = all_specs()
        self.assertEqual(len(specs), 8)
        self.assertEqual(len({s.name for s in specs}), 8)
        for spec in specs:
            net = build_network(spec)
            self.assertEqual(net["name"], spec.name)
            self.assertEqual(NetworkSpec.from_name(spec.name), spec)
            tasks = [next(iter(t)) for t in net["pipeline"]]
            self.assertEqual(tasks[:2], [TASK_PERSONS, TASK_TRACKS])
            self.assertEqual(TASK_FACES in tasks, spec.face)
            self.assertEqual(TASK_FACE_EMB in tasks, spec.face)
            self.assertEqual(TASK_PLATES in tasks, spec.lpr)
            self.assertEqual(TASK_PLATE_TEXT in tasks, spec.lpr)
            # every referenced model is defined, and AIPU cores never exceed the 4 Metis cores
            for t in net["pipeline"]:
                self.assertIn(next(iter(t.values()))["model_name"], net["models"])
            cores = [m["extra_kwargs"]["aipu_cores"] for k, m in net["models"].items() if k != "tracker"]
            self.assertTrue(all(c >= 1 for c in cores))
            self.assertLessEqual(sum(cores), max(4, len(cores)))
            for m in net["models"].values():
                if "dataset" in m:
                    self.assertIn(m["dataset"], net["datasets"])

    def test_yaml_round_trip_and_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = os.path.join(tmp, "cal")
            os.makedirs(cal)
            opts = BuildOptions(calibration_dir=cal, person_conf=0.33)
            path = write_network(NetworkSpec("yolov8spose", face=True, lpr=True), tmp, opts)
            text = open(path).read()
            self.assertIn("- torch-totensor:\n", text)  # bare operator keys, Voyager style
            data = yaml.safe_load(text)
            self.assertEqual(data["datasets"]["CocoDataset-keypoint-COCO2017"]["repr_imgs_dir_path"],
                             os.path.realpath(cal))
            self.assertNotIn("repr_imgs_dir_path", data["datasets"]["LFWTorchvisionPeople"])
            dec = data["pipeline"][0][TASK_PERSONS]["postprocess"][0]["decodeyolopose"]
            self.assertAlmostEqual(dec["conf_threshold"], 0.33)

    def test_enroll_network(self):
        net = face_enroll_spec_yaml()
        self.assertEqual([next(iter(t)) for t in net["pipeline"]], [TASK_FACES, TASK_FACE_EMB])
        self.assertEqual(len(net["models"]), 2)


class TestCapacityPlanner(unittest.TestCase):

    def test_tile_count(self):
        self.assertEqual(tile_count((640, 360), 10), 1)
        self.assertEqual(tile_count((1920, 1080), 10), 4 * 2 + 1)

    def test_degrades_monotonically_with_camera_count(self):
        planner = CapacityPlanner()
        for feats in (frozenset(), frozenset({"face"}), frozenset({"face", "lpr"})):
            prev_rank, prev_fps = -1, 999
            for n in range(1, 16):
                plan = planner.plan(_cams(n, feats))
                rank = TIER_RANK[plan.tier.name]
                self.assertGreaterEqual(rank, prev_rank, f"{feats} n={n}")
                if rank == prev_rank:
                    self.assertLessEqual(plan.fps, prev_fps)
                prev_rank, prev_fps = rank, plan.fps
                self.assertLessEqual(plan.aipu_utilization, planner.budgets.target_utilization + 1e-6)
                self.assertFalse(plan.overloaded, f"{feats} n={n}")
                self.assertGreaterEqual(plan.fps, plan.tier.min_fps)

    def test_one_camera_gets_top_quality(self):
        # Tiling is opt-in (native abort on SDK 1.6.0), so the default top tier is `high`.
        plan = CapacityPlanner().plan(_cams(1))
        self.assertEqual(plan.tier.name, "high")
        self.assertEqual(plan.fps, 25)
        self.assertEqual(plan.groups[0].sources[0], "rtsp://cam0/main")
        self.assertEqual(plan.groups[0].network, "upside-yolov8spose")

    def test_one_camera_tiled_when_opted_in(self):
        from src.voyager.capacity_planner import CapacityBudgets

        plan = CapacityPlanner(CapacityBudgets(enable_tiling=True)).plan(_cams(1))
        self.assertEqual(plan.tier.name, "ultra")
        self.assertEqual(plan.fps, 25)
        self.assertTrue(plan.groups[0].sources[0].startswith("tile[size=640,overlap=10]:rtsp://cam0/main"))
        self.assertEqual(plan.groups[0].network, "upside-yolov8spose")

    def test_tracker_profile(self):
        plan = CapacityPlanner().plan(_cams(1), profile="tracker")
        self.assertEqual(plan.tier.name, "tracker")
        self.assertEqual(plan.tier.pose_model, "yolov5m-v7-coco-tracker")
        self.assertEqual(plan.groups[0].network, "yolov5m-v7-coco-tracker")
        self.assertEqual(plan.groups[0].sources[0], "rtsp://cam0/sub")

    def test_tracker_profile_with_pose_feature_upgrades(self):
        plan = CapacityPlanner().plan(_cams(1, frozenset({"pose"})), profile="tracker")
        self.assertNotEqual(plan.tier.name, "tracker")
        self.assertIn(plan.tier.pose_model, ("yolov8npose", "yolov8spose"))

    def test_many_cameras_with_faces_scale_down(self):
        planner = CapacityPlanner()
        plan15 = planner.plan(_cams(15, frozenset({"face"})))
        self.assertGreater(TIER_RANK[plan15.tier.name], TIER_RANK["high"])
        self.assertEqual(plan15.tier.pose_model, "yolov8npose")
        self.assertTrue(all(s.startswith("rtsp://") and "/sub" in s for s in plan15.groups[0].sources))

    def test_decode_budget_forces_sub_stream(self):
        b = CapacityBudgets(decode_mpx_per_s=300)
        plan = CapacityPlanner(b).plan(_cams(6))
        self.assertEqual(plan.tier.stream, "sub")

    def test_groups_by_feature_set(self):
        cams = _cams(3)
        cams[1].features = frozenset({"face"})
        plan = CapacityPlanner().plan(cams)
        names = {g.network: g.camera_ids for g in plan.groups}
        self.assertEqual(len(names), 2)
        self.assertTrue(any(n.endswith("-face") and ids == ["cam_01"] for n, ids in names.items()))

    def test_fixed_profile_and_overload_flag(self):
        planner = CapacityPlanner()
        plan = planner.plan(_cams(2), profile="economy")
        self.assertEqual(plan.tier.name, "economy")
        self.assertLessEqual(plan.fps, 10)
        heavy = planner.plan(_cams(60, frozenset({"face", "lpr"})))
        self.assertTrue(heavy.overloaded)
        self.assertTrue(heavy.reasons)
        with self.assertRaises(ValueError):
            planner.plan(_cams(1), profile="bogus")


class _Obj:
    def __init__(self, sec):
        self._sec = sec

    def get_secondary_meta(self, task):
        return self._sec.get(task)


class TestResultParser(unittest.TestCase):

    def test_geometry(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        m = iou_matrix(a, a)
        self.assertAlmostEqual(float(m[0, 0]), 1.0)
        self.assertAlmostEqual(float(m[0, 1]), 0.0)
        persons = np.array([[0, 0, 100, 300], [200, 0, 300, 300]], dtype=np.float32)
        faces = np.array([[40, 10, 60, 40], [240, 200, 260, 230], [500, 0, 510, 10]], dtype=np.float32)
        self.assertEqual(assign_faces_to_persons(faces, persons).tolist(), [0, -1, -1])

    def test_parse_full_frame(self):
        emb = np.random.randn(512).astype(np.float32)
        meta = {
            TASK_PERSONS: SimpleNamespace(boxes=np.array([[0, 0, 100, 300], [200, 0, 300, 300]], np.float32),
                                          scores=np.array([0.9, 0.8], np.float32),
                                          keypoints=np.ones((2, 17, 3), np.float32)),
            TASK_TRACKS: SimpleNamespace(boxes={7: [201, 1, 299, 299], 3: [1, 0, 99, 301]}),
            TASK_FACES: SimpleNamespace(boxes=np.array([[40, 10, 60, 40]], np.float32),
                                        scores=np.array([0.95], np.float32),
                                        objects=[_Obj({TASK_FACE_EMB: SimpleNamespace(embedding=[emb.tolist()])})]),
            TASK_PLATES: SimpleNamespace(boxes=np.array([[5, 5, 50, 20]], np.float32), scores=np.array([0.7]),
                                         objects=[_Obj({TASK_PLATE_TEXT: SimpleNamespace(label=" AB123 ")})]),
        }
        img = SimpleNamespace(width=1920, height=1080)
        fa = parse_frame_result(SimpleNamespace(meta=meta, image=img, stream_id=4), 1.0)
        self.assertEqual(fa.stream_id, 4)
        self.assertEqual(fa.frame_size, (1920, 1080))
        self.assertEqual([p["track_id"] for p in fa.persons], [3, 7])
        self.assertEqual(fa.persons[0]["keypoints"].shape, (17, 3))
        self.assertEqual(fa.faces[0]["person_index"], 0)
        np.testing.assert_allclose(fa.faces[0]["embedding"], emb, rtol=1e-6)
        self.assertEqual(fa.plates[0]["plate_text"], "AB123")

    def test_parse_pose_only(self):
        meta = {TASK_PERSONS: SimpleNamespace(boxes=np.zeros((0, 4)), scores=np.zeros(0), keypoints=np.zeros((0, 17, 3)))}
        fa = parse_frame_result(SimpleNamespace(meta=meta, image=None, stream_id=0), 0.0)
        self.assertEqual(fa.persons, [])
        self.assertEqual(fa.faces, [])

    def test_parse_pedestrian_and_vehicle_tracker(self):
        # Native tracker object (matches yolov5m-v7-coco-tracker / upside_remote_cross_line_monitor.py)
        class MockTrack:
            def __init__(self, track_id, class_id, history, score=0.92):
                self.track_id = track_id
                self.class_id = class_id
                self.history = history
                self.score = score

        tracks = [
            MockTrack(101, 0, np.array([[10, 20, 80, 150], [12, 22, 82, 152]])),  # person
            MockTrack(102, 2, np.array([[200, 300, 400, 500]])),  # car (class_id=2, ignored)
            MockTrack(103, 0, np.array([[50, 60, 100, 180]])),  # person
        ]
        res = SimpleNamespace(
            meta={},
            stream_id=1,
            image=SimpleNamespace(width=1280, height=720),
            pedestrian_and_vehicle_tracker=tracks,
        )
        fa = parse_frame_result(res, 2.0)
        self.assertEqual(fa.stream_id, 1)
        self.assertEqual(fa.frame_size, (1280, 720))
        self.assertEqual(len(fa.persons), 2)
        self.assertEqual(fa.persons[0]["track_id"], 101)
        self.assertEqual(fa.persons[0]["bbox"], [12, 22, 82, 152])
        self.assertEqual(fa.persons[0]["confidence"], 0.92)
        self.assertEqual(fa.persons[1]["track_id"], 103)
        self.assertEqual(fa.persons[1]["bbox"], [50, 60, 100, 180])

    def test_parse_keypoints_with_tracker_association(self):
        class MockTrack:
            def __init__(self, track_id, class_id, history, score=0.9):
                self.track_id = track_id
                self.class_id = class_id
                self.history = history
                self.score = score

        tracks = [
            MockTrack(10, 0, np.array([[10, 10, 100, 200]])),
        ]
        # Keypoints detected for the person at [10, 10, 100, 200]
        kpts = np.full((1, 17, 3), 42.0, dtype=np.float32)
        meta = {
            "keypoint_detections": SimpleNamespace(
                boxes=np.array([[10, 10, 100, 200]], dtype=np.float32),
                scores=np.array([0.95], dtype=np.float32),
                keypoints=kpts,
            )
        }
        res = SimpleNamespace(
            meta=meta,
            stream_id=0,
            image=None,
            pedestrian_and_vehicle_tracker=tracks,
        )
        fa = parse_frame_result(res, 1.0)
        self.assertEqual(len(fa.persons), 1)
        self.assertEqual(fa.persons[0]["track_id"], 10)
        self.assertIsNotNone(fa.persons[0]["keypoints"])
        self.assertEqual(fa.persons[0]["keypoints"].shape, (17, 3))
        self.assertEqual(fa.persons[0]["keypoints"][0, 0], 42.0)

    def test_parse_keypoints_from_various_meta_formats(self):
        # 2D (17, 2) keypoints without confidence -> auto adds 1.0 confidence column
        kpts_2d = np.ones((1, 17, 2), dtype=np.float32) * 15.0
        meta = {
            "pose": SimpleNamespace(
                boxes=np.array([[5, 5, 50, 100]], dtype=np.float32),
                scores=np.array([0.88], dtype=np.float32),
                keypoints=kpts_2d,
            )
        }
        res = SimpleNamespace(meta=meta, image=None, stream_id=0)
        fa = parse_frame_result(res, 1.0)
        self.assertEqual(len(fa.persons), 1)
        self.assertEqual(fa.persons[0]["keypoints"].shape, (17, 3))
        self.assertEqual(fa.persons[0]["keypoints"][0, 2], 1.0)


class TestFaceDatabase(unittest.TestCase):

    def test_enroll_match_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FaceDatabase(db_path=os.path.join(tmp, "db.json"))
            john = np.random.randn(512).astype(np.float32)
            jane = np.random.randn(512).astype(np.float32)
            db.add_identity("John", john)
            db.add_identity("Jane", jane)
            res = db.match_batch(np.stack([john, jane, np.random.randn(512)]), threshold=0.9)
            self.assertEqual([r[0] for r in res], ["John", "Jane", "Unknown"])
            self.assertEqual(db.match(john, 0.5)[0], "John")
            db2 = FaceDatabase(db_path=os.path.join(tmp, "db.json"))
            self.assertEqual(sorted(db2.list_identities()), ["Jane", "John"])
            self.assertTrue(db2.remove_identity("Jane"))

    def test_legacy_file_is_backed_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "db.json")
            import json
            with open(path, "w") as f:
                json.dump({"Old": [0.1] * 512}, f)  # legacy v1 flat format
            db = FaceDatabase(db_path=path)
            self.assertEqual(db.list_identities(), [])
            self.assertTrue(os.path.exists(os.path.join(tmp, "db.legacy.json")))


class TestIdentityResolver(unittest.TestCase):

    def test_votes_stabilise(self):
        r = TrackIdentityResolver()
        name = None
        for _ in range(3):
            name, _ = r.update("c", 1, "Alice", 0.8, now=1.0)
        self.assertEqual(name, "Alice")
        name, _ = r.update("c", 1, "Bob", 0.6, now=1.1)  # single outlier doesn't flip identity
        self.assertEqual(name, "Alice")
        r.purge("c", [], now=100.0)
        self.assertIsNone(r.get("c", 1))


class TestPipelineSimulated(unittest.TestCase):

    def _wait(self, cond, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return True
            time.sleep(0.05)
        return False

    def test_pipeline_runs_and_replans(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = MultiCameraPipeline(_sim_config(tmp, 2), simulate=True)
            p.start()
            try:
                self.assertEqual(p.runtime.backend, "simulation")
                self.assertTrue(self._wait(lambda: p.get_stats()["persons"] > 0))
                self.assertEqual(p.plan.groups[0].network, "upside-yolov8spose-face")
                cams = p.get_camera_status()
                self.assertEqual(len(cams), 2)
                self.assertTrue(self._wait(lambda: all(c["state"] == "live" for c in p.get_camera_status())))

                # Disabling face recognition changes the network composition
                p.set_model_enabled("face_recognizer", False)
                p.replan_now()
                self.assertTrue(all(not g.spec.face for g in p.plan.groups))

                # Per-camera override splits cameras into feature groups
                p.set_model_enabled("face_recognizer", True, "cam_01")
                p.replan_now()
                self.assertEqual(len(p.plan.groups), 2)

                # Adding cameras re-plans to a lower tier
                for i in range(2, 12):
                    p.add_camera({"id": f"cam_{i:02d}", "source": f"rtsp://cam{i}/main",
                                  "source_sub": f"rtsp://cam{i}/sub"})
                p.set_model_enabled("face_recognizer", True)
                p.replan_now()
                self.assertGreater(TIER_RANK[p.plan.tier.name], TIER_RANK["ultra"])
                self.assertTrue(p.remove_camera("cam_11"))
                self.assertEqual(len(p.get_camera_status()), 11)
            finally:
                p.stop()

    def test_live_enrollment_and_recognition(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sim_config(tmp, 1)
            p = MultiCameraPipeline(cfg, simulate=True)
            p.start()
            try:
                self.assertTrue(self._wait(lambda: p._latest.get("cam_00", {}).get("faces")))
                ok, msg = p.enroll_identity("Alice", cam_id="cam_00")
                self.assertTrue(ok, msg)
                self.assertTrue(self._wait(lambda: p.get_stats()["faces_recognized"] >= 1))
            finally:
                p.stop()

    def test_preview_only_renders_when_viewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = MultiCameraPipeline(_sim_config(tmp, 2), simulate=True)
            p.start()
            try:
                time.sleep(0.5)
                self.assertIsNone(p.preview.latest("cam_00"))
                p.preview.acquire("cam_00")
                item = p.preview.wait_next("cam_00", -1, timeout=5)
                self.assertIsNotNone(item)
                self.assertTrue(item[1][:2] == b"\xff\xd8")  # JPEG SOI
                p.preview.release("cam_00")
                self.assertIsNotNone(p.preview.snapshot("grid", timeout=5))
            finally:
                p.stop()

    def test_visualizer(self):
        vis = Visualizer({})
        frame = np.zeros((360, 640, 3), np.uint8)
        out = vis.draw_frame(frame, detections=[{"bbox": [10, 10, 100, 200], "confidence": 0.9, "track_id": 1}],
                             poses=[{"keypoints": np.full((17, 3), 50.0)}],
                             faces=[{"bbox": [20, 20, 60, 60], "name": "Alice", "similarity": 0.8}],
                             stream_title="t", fps=10.0)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)

    def test_visualizer_normalized_and_pixel_keypoints(self):
        vis = Visualizer({})
        frame = np.zeros((360, 640, 3), np.uint8)
        # Normalized keypoints in [0.0, 1.0]
        kpts_norm = np.array([
            [0.5, 0.1, 0.9], [0.4, 0.15, 0.9], [0.6, 0.15, 0.9],
            [0.3, 0.2, 0.9], [0.7, 0.2, 0.9], [0.4, 0.3, 0.9],
            [0.6, 0.3, 0.9], [0.35, 0.4, 0.9], [0.65, 0.4, 0.9],
            [0.3, 0.5, 0.9], [0.7, 0.5, 0.9], [0.45, 0.6, 0.9],
            [0.55, 0.6, 0.9], [0.45, 0.75, 0.9], [0.55, 0.75, 0.9],
            [0.45, 0.9, 0.9], [0.55, 0.9, 0.9],
        ], dtype=np.float32)
        out = vis.draw_frame(frame, detections=[], poses=[{"keypoints": kpts_norm}],
                             faces=[], stream_title="test", fps=15.0)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)

    def test_compose_grid_aspect_ratio(self):
        from src.render.preview import PreviewHub
        hub = PreviewHub(lambda f, p: f, grid_size=(1280, 720))
        try:
            # 2 cameras, 16:9 frames (640x360)
            f1 = np.zeros((360, 640, 3), dtype=np.uint8)
            f2 = np.zeros((360, 640, 3), dtype=np.uint8)
            hub._tiles = {"cam1": f1, "cam2": f2}
            grid = hub.compose_grid()
            # 2 cameras side by side: each tile is 640x360 -> total size 1280x360 (16:9 each)
            self.assertEqual(grid.shape[1], 1280)
            self.assertEqual(grid.shape[0], 360)
            self.assertAlmostEqual((grid.shape[1] / 2) / grid.shape[0], 16.0 / 9.0, places=2)
        finally:
            hub.stop()


class TestWebApi(unittest.TestCase):

    def test_api_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            cfg = _sim_config(tmp, 2)
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)
            p = MultiCameraPipeline(cfg, simulate=True, config_path=cfg_path)
            p.start()
            srv = WebServer(p, host="127.0.0.1", port=0, use_https=False)
            try:
                for method, path, body in [("GET", "/api/status", {}), ("GET", "/api/models", {}),
                                           ("GET", "/api/capacity", {}), ("GET", "/api/cameras", {})]:
                    status, payload = srv.handle_api(method, path, {}, body)
                    self.assertEqual(status, 200, path)
                st = srv.handle_api("GET", "/api/status", {}, {})[1]
                self.assertEqual(st["backend"], "simulation")
                self.assertIn("aipu", st["telemetry"])
                self.assertEqual(len(st["models"]), 5)

                status, payload = srv.handle_api("POST", "/api/models/toggle", {},
                                                 {"model_id": "human_detector", "enabled": False})
                self.assertEqual(status, 200)
                self.assertFalse(p.is_model_enabled("human_detector"))
                self.assertEqual(srv.handle_api("POST", "/api/models/toggle", {}, {"model_id": "nope"})[0], 400)

                status, payload = srv.handle_api("POST", "/api/capacity/profile", {}, {"profile": "economy"})
                self.assertEqual(payload["profile"], "economy")
                # legacy dashboard profile names still work
                status, payload = srv.handle_api("POST", "/api/models/configure_detector", {},
                                                 {"profile": "high_precision", "detect_interval": 2})
                self.assertEqual(payload["profile"], "high")

                status, payload = srv.handle_api("POST", "/api/cameras", {}, {"id": "cam_x", "source": "rtsp://x"})
                self.assertEqual(status, 200)
                with open(cfg_path) as f:
                    self.assertIn("cam_x", [c["id"] for c in yaml.safe_load(f)["cameras"]])
                self.assertEqual(srv.handle_api("POST", "/api/cameras", {}, {"id": "cam_x", "source": "rtsp://x"})[0], 400)
                self.assertEqual(srv.handle_api("DELETE", "/api/cameras/cam_x", {}, {})[0], 200)

                status, payload = srv.handle_api("POST", "/api/enroll_face", {}, {"name": "Bob"})
                self.assertEqual(status, 400)
                img = np.random.randint(0, 255, (120, 120, 3), np.uint8)

                # Test enrolling via image even when face_recognizer is disabled
                p.set_model_enabled("face_recognizer", False)
                status, payload = srv.handle_api("POST", "/api/enroll_face", {}, {"name": "Bob", "_image": img})
                self.assertEqual(status, 200, payload)
                self.assertIn("Bob", srv.handle_api("GET", "/api/identities", {}, {})[1]["identities"])

                # Test use_live_frame requires face_recognizer to be on
                status, payload = srv.handle_api("POST", "/api/enroll_face", {}, {"name": "Charlie", "use_live_frame": True})
                self.assertEqual(status, 400)
                self.assertIn("Face recognition must be enabled", payload.get("error", ""))

                # Re-enable face_recognizer and test live frame enrollment
                p.set_model_enabled("face_recognizer", True)
                with p._state_lock:
                    p._latest.setdefault("cam_00", {})
                    p._latest["cam_00"].update({
                        "persons": 1, "faces_detected": 1, "known_faces": 0, "plates": 0,
                        "faces": [{"bbox": [10, 10, 100, 100], "name": "Unknown", "embedding": np.ones(512)}]
                    })
                status, payload = srv.handle_api("POST", "/api/enroll_face", {}, {"name": "Charlie", "use_live_frame": True, "camera_id": "cam_00"})
                self.assertEqual(status, 200, payload)
                self.assertIn("Charlie", srv.handle_api("GET", "/api/identities", {}, {})[1]["identities"])

                # display & system info endpoints
                status, disp = srv.handle_api("GET", "/api/display", {}, {})
                self.assertEqual(status, 200)
                self.assertIn("draw_boxes", disp)
                status, disp_post = srv.handle_api("POST", "/api/display", {}, {"draw_boxes": False})
                self.assertEqual(status, 200)
                self.assertFalse(disp_post["config"]["draw_boxes"])

                status, sys_info = srv.handle_api("GET", "/api/system_info", {}, {})
                self.assertEqual(status, 200)
                self.assertEqual(sys_info["host"]["product_name"], "Upside AI Box")

                # logs endpoints
                status, logs_list = srv.handle_api("GET", "/api/logs/list", {}, {})
                self.assertEqual(status, 200)
                self.assertIn("logs", logs_list)
                self.assertIn("retention_days", logs_list)

                status, logs_view = srv.handle_api("GET", "/api/logs/view", {"filename": logs_list.get("current")}, {})
                self.assertEqual(status, 200)
                self.assertIn("content", logs_view)

                if srv.app:
                    client = srv.app.test_client()
                    self.assertEqual(client.get("/").status_code, 200)
                    self.assertEqual(client.get("/settings").status_code, 200)
                    self.assertEqual(client.get("/api/status").status_code, 200)
                    self.assertEqual(client.get("/api/logs/list").status_code, 200)
                    self.assertEqual(client.get("/api/logs/download").status_code, 200)
                    self.assertEqual(client.post("/api/models/toggle",
                                                 json={"model_id": "pose_estimator", "enabled": False}).status_code, 200)
                    self.assertFalse(p.is_model_enabled("pose_estimator"))
                    self.assertEqual(client.get("/api/nope").status_code, 404)
            finally:
                p.stop()

    def test_auto_updater_and_web_api(self):
        """Tests AutoUpdater configuration, intervals, git check mocks, and WebServer update endpoints."""
        from unittest.mock import patch
        from src.utils.auto_updater import AutoUpdater, CHECK_INTERVALS_SEC

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_cfg_path = os.path.join(tmpdir, "test_config.yaml")
            initial_cfg = _sim_config(tmpdir, 0)
            initial_cfg["auto_update"] = {"enabled": True, "branch": "live", "check_interval": "daily",
                                          "install_time": "immediately", "remote": "origin"}
            with open(tmp_cfg_path, "w") as f:
                yaml.safe_dump(initial_cfg, f)

            updater = AutoUpdater(initial_cfg, config_path=tmp_cfg_path, restart_callback=lambda: None)
            self.assertTrue(updater.enabled)
            self.assertEqual(CHECK_INTERVALS_SEC["daily"], 86400)
            updater.update_config({"enabled": False, "check_interval": "weekly", "install_time": "03:00",
                                   "branch": "release-v1"})
            with open(tmp_cfg_path, "r") as f:
                saved_yaml = yaml.safe_load(f)
            self.assertFalse(saved_yaml["auto_update"]["enabled"])
            self.assertEqual(saved_yaml["auto_update"]["check_interval"], "weekly")

            pipeline = MultiCameraPipeline(initial_cfg, simulate=True)
            web_server = WebServer(pipeline, auto_updater=updater, host="127.0.0.1", port=8000, use_https=False)
            status, data = web_server.handle_api("GET", "/api/update/status", {}, {})
            self.assertEqual(data["branch"], "release-v1")
            web_server.handle_api("POST", "/api/update/config", {}, {"enabled": True, "check_interval": "hourly"})
            self.assertTrue(updater.enabled)
            with patch.object(updater, "check_for_updates", return_value={"success": True}):
                self.assertTrue(web_server.handle_api("POST", "/api/update/check", {}, {})[1]["success"])
            with patch.object(updater, "apply_update_and_restart", return_value={"success": True}):
                self.assertTrue(web_server.handle_api("POST", "/api/update/install", {}, {})[1]["success"])
            updater.stop()
            pipeline.preview.stop()

    def test_voyager_runtime_pipeline_kwargs(self):
        """Verifies that VoyagerRuntime defaults specified_frame_rate to 0 to prevent videorate duplication."""
        from unittest.mock import patch
        from src.voyager.stream_runtime import VoyagerRuntime
        from src.voyager.capacity_planner import CapacityPlan, TIERS

        with patch("src.voyager.sdk.sdk_modules", return_value=SimpleNamespace()):
            runtime = VoyagerRuntime({"low_latency": False, "aipu_cores": 4, "rtsp_latency_ms": 200},
                                     lambda r: None, lambda *a: None)
            dummy_plan = SimpleNamespace(fps=25)
            kwargs = runtime._pipeline_kwargs(dummy_plan)
            self.assertEqual(kwargs["specified_frame_rate"], 0)
            self.assertFalse(kwargs["low_latency"])
            self.assertEqual(kwargs["aipu_cores"], 4)

            # Explicit override
            runtime_custom = VoyagerRuntime({"specified_frame_rate": 15}, lambda r: None, lambda *a: None)
            self.assertEqual(runtime_custom._pipeline_kwargs(dummy_plan)["specified_frame_rate"], 15)

    def test_diag_hw_decode_helpers(self):
        """Verifies hardware video decoding diagnostic helper functions execute reliably."""
        from tools.diag_hw_decode import (
            calculate_cpu_percent,
            check_config_yaml,
            check_hardware_nodes,
            check_system,
        )

        check_system()
        nodes = check_hardware_nodes()
        self.assertIn("/dev/mpp_service", nodes)
        self.assertIn("/dev/rga", nodes)

        pct = calculate_cpu_percent((100.0, 50.0), (200.0, 75.0))
        self.assertAlmostEqual(pct, 75.0)

        # calculate_cpu_percent with zero delta
        self.assertEqual(calculate_cpu_percent((100.0, 50.0), (100.0, 50.0)), 0.0)

        # check_config_yaml reads camera stream
        cam_url = check_config_yaml()
        self.assertTrue(cam_url is None or cam_url.startswith("rtsp://"))

    def test_daily_rolling_logger_and_retention(self):
        """Verifies daily rolling logger creation, record capture, line reading, and 30-day retention pruning."""
        from src.utils.logger import DailyRollingLogger
        import datetime

        with tempfile.TemporaryDirectory() as tmp_log_dir:
            logger = DailyRollingLogger(log_dir=tmp_log_dir, max_retention_days=30, capture_stdout=False)
            logger.write_record("Line 1: System started.\n")
            logger.write_record("Line 2: Metis AIPU online.\n")

            logs = logger.list_logs()
            self.assertEqual(len(logs), 1)
            self.assertTrue(logs[0]["is_current"])

            # Read log
            res = logger.read_log(logs[0]["filename"], tail=10)
            self.assertIn("Metis AIPU online", res["content"])
            self.assertEqual(res["total_lines"], 2)

            # Create mock old log beyond 30 days retention
            old_date = (datetime.date.today() - datetime.timedelta(days=35)).isoformat()
            old_file = os.path.join(tmp_log_dir, f"app_{old_date}.log")
            with open(old_file, "w") as f:
                f.write("Old log message\n")

            # Create mock recent log within 30 days
            recent_date = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
            recent_file = os.path.join(tmp_log_dir, f"app_{recent_date}.log")
            with open(recent_file, "w") as f:
                f.write("Recent log message\n")

            self.assertEqual(len(logger.list_logs()), 3)

            # Run cleanup
            removed = logger.cleanup_old_logs(max_days=30)
            self.assertIn(f"app_{old_date}.log", removed)
            self.assertFalse(os.path.exists(old_file))
            self.assertTrue(os.path.exists(recent_file))

            # Verify list_logs only has 2 files now
            remaining = [l["filename"] for l in logger.list_logs()]
            self.assertIn(f"app_{recent_date}.log", remaining)
            self.assertNotIn(f"app_{old_date}.log", remaining)

    def test_system_config_manager_and_first_launch(self):
        """Verifies system-specific configuration files in ~/.upside-ai/config on first launch and preservation."""
        from src.utils.config_manager import ensure_system_configs, resolve_active_config_path

        with tempfile.TemporaryDirectory() as tmp_home:
            # 1. First launch: ensure_system_configs should create tmp_home/config and copy config.yaml
            active_cfg = ensure_system_configs(project_config_dir="config", custom_home=tmp_home)
            self.assertTrue(os.path.isfile(active_cfg))
            self.assertEqual(active_cfg, os.path.join(tmp_home, "config", "config.yaml"))

            with open(active_cfg, "r") as f:
                loaded = yaml.safe_load(f)
            self.assertIn("cameras", loaded)

            # 2. Modify config in system config dir
            loaded["web"]["port"] = 9999
            with open(active_cfg, "w") as f:
                yaml.safe_dump(loaded, f)

            # 3. Subsequent launch: ensure_system_configs must NOT overwrite customized config
            active_cfg_2 = ensure_system_configs(project_config_dir="config", custom_home=tmp_home)
            with open(active_cfg_2, "r") as f:
                loaded_2 = yaml.safe_load(f)
            self.assertEqual(loaded_2["web"]["port"], 9999)

            # 4. resolve_active_config_path with None -> returns system config
            resolved = resolve_active_config_path(cli_config_arg=None, project_config_dir="config", custom_home=tmp_home)
            self.assertEqual(resolved, active_cfg)

            # 5. resolve_active_config_path with custom file
            custom_file = os.path.join(tmp_home, "custom.yaml")
            with open(custom_file, "w") as f:
                yaml.safe_dump({"custom": True}, f)
            resolved_custom = resolve_active_config_path(cli_config_arg=custom_file, project_config_dir="config", custom_home=tmp_home)
            self.assertEqual(resolved_custom, os.path.abspath(custom_file))

    def test_settings_persist_to_active_config_and_system_info(self):
        """Verifies changes made in settings save into the active config file and system_info exposes paths."""
        from src.pipeline import MultiCameraPipeline
        from src.web.server import WebServer, get_system_info

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = os.path.join(tmp_dir, "config.yaml")
            with open("config/config.yaml", "r") as f:
                orig_cfg = yaml.safe_load(f)
            with open(cfg_file, "w") as f:
                yaml.safe_dump(orig_cfg, f)

            pipeline = MultiCameraPipeline(orig_cfg, simulate=True, config_path=cfg_file)
            server = WebServer(pipeline, config_path=cfg_file)

            # Verify system_info payload includes config_path and logs_dir
            status, info = server.handle_api("GET", "/api/system_info", {}, {})
            self.assertEqual(status, 200)
            self.assertEqual(info["host"]["config_path"], cfg_file)
            self.assertIn(".upside-ai/logs", info["host"]["logs_dir"])

            # Verify settings change (e.g. model toggle) persists into cfg_file
            status, res = server.handle_api("POST", "/api/models/toggle", {}, {"model_id": "human_detector", "enabled": False})
            self.assertEqual(status, 200)

            with open(cfg_file, "r") as f:
                persisted = yaml.safe_load(f)
            self.assertFalse(persisted["models"]["human_detector"]["enabled"])

    def test_system_restart_api(self):
        """Verifies /api/system/restart triggers graceful restart response."""
        import threading
        from src.pipeline import MultiCameraPipeline
        from src.web.server import WebServer
        from src.utils.auto_updater import AutoUpdater

        restarted = threading.Event()
        def mock_restart():
            restarted.set()

        pipeline = MultiCameraPipeline({"cameras": []}, simulate=True)
        updater = AutoUpdater({"auto_update": {"enabled": False}}, restart_callback=mock_restart)
        server = WebServer(pipeline, auto_updater=updater)

        status, payload = server.handle_api("POST", "/api/system/restart", {}, {})
        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "restarting")

        # Wait for deferred restart thread to execute callback
        self.assertTrue(restarted.wait(timeout=2.5))


if __name__ == "__main__":
    unittest.main()

