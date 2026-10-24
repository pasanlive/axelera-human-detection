"""
WebServer: Low-Latency MJPEG Web Streaming and REST API for Axelera Metis System.
Supports HTTPS SSL encryption, Flask, and native Python http.server fallback for secure remote local network access.

All REST endpoints are implemented once in :meth:`WebServer.handle_api` and served
by either Flask or the zero-dependency native server. Video previews are rendered
on demand by the pipeline's PreviewHub: a camera is only annotated/encoded while
at least one browser is watching it.
"""

import base64
import json
import os
import socket
import ssl
import subprocess
import threading
import time
import urllib.parse
from typing import Any, Dict, Generator, Optional, Tuple

import cv2
import numpy as np

# Optional Flask support
try:
    from flask import Flask, Response, jsonify, render_template, request
    try:
        from flask_cors import CORS
        FLASK_CORS_AVAILABLE = True
    except ImportError:
        FLASK_CORS_AVAILABLE = False
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Fallback Python standard library HTTP server
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

AIPU_TOPS_MAX = 214.0
AIPU_CORES = 4


def get_local_ip() -> str:
    """Returns local network IP address of the host system."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_hardware_backend_name(pipeline) -> str:
    """Describes the active inference backend."""
    backend = getattr(getattr(pipeline, "runtime", None), "backend", None)
    if backend == "axelera_voyager":
        return "Axelera Metis AIPU (Voyager SDK, 4 Cores)"
    if backend == "simulation":
        return "SIMULATION (no AIPU inference)"
    return "Inference backend unavailable"


def _host_telemetry() -> Dict[str, Any]:
    ram_used_gb = ram_total_gb = ram_percent = cpu_percent = 0.0
    temp_c = None
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_used_gb = round(mem.used / (1024 ** 3), 2)
        ram_total_gb = round(mem.total / (1024 ** 3), 2)
        ram_percent = round(mem.percent, 1)
        cpu_percent = round(psutil.cpu_percent(interval=None), 1)
        if hasattr(psutil, "sensors_temperatures"):
            for entries in (psutil.sensors_temperatures() or {}).values():
                if entries and entries[0].current > 0:
                    temp_c = round(entries[0].current, 1)
                    break
    except Exception:
        pass
    if temp_c is None:
        for zone_id in range(4):
            path = f"/sys/class/thermal/thermal_zone{zone_id}/temp"
            try:
                with open(path) as f:
                    val = float(f.read().strip())
                temp_c = round(val / 1000.0, 1) if val > 1000 else round(val, 1)
                break
            except Exception:
                continue
    return {"ram": {"used_gb": ram_used_gb, "total_gb": ram_total_gb, "percent": ram_percent},
            "cpu_percent": cpu_percent, "host_temperature_c": temp_c}


def get_system_telemetry(pipeline=None) -> Dict[str, Any]:
    """
    Gathers real-time system & hardware telemetry:
    - RAM / CPU usage of the RK3588 host
    - Metis core temperature and per-core device FPS from Voyager tracers
    - AIPU load: planned utilisation scaled by achieved vs planned throughput
      (Voyager does not expose a direct utilisation counter; marked as ``estimated``)
    """
    out = _host_telemetry()
    metrics = {}
    plan = None
    stats = {}
    if pipeline is not None:
        try:
            metrics = pipeline.get_runtime_metrics() or {}
            plan = pipeline.plan
            stats = pipeline.get_stats()
        except Exception:
            pass

    core_temp = metrics.get("core_temp", {}).get("value")
    out["temperature_c"] = round(core_temp, 1) if core_temp else out.get("host_temperature_c")
    out["temperature_source"] = "metis_core" if core_temp else "host"

    running = bool(getattr(pipeline, "is_running", False))
    load = 0.0
    if running and plan is not None and plan.fps > 0:
        planned_total = plan.fps * sum(len(g.camera_ids) for g in plan.groups)
        achieved = float(stats.get("total_fps", 0.0))
        ratio = min(1.2, achieved / planned_total) if planned_total else 0.0
        load = round(min(100.0, plan.aipu_utilization * 100.0 * ratio), 1)

    per_core = []
    for c in range(AIPU_CORES):
        dev_fps = None
        for key in (f"device_d0_{c}_fps", f"device_{c}_fps"):
            if key in metrics:
                dev_fps = metrics[key]["value"]
                break
        per_core.append({
            "core_id": c, "name": f"Core {c}", "load_percent": load,
            "tops_current": round(AIPU_TOPS_MAX / AIPU_CORES * load / 100.0, 1),
            "tops_max": AIPU_TOPS_MAX / AIPU_CORES, "device_fps": dev_fps, "active": load > 0,
        })
    out["aipu"] = {
        "cores": AIPU_CORES,
        "load_percent": load,
        "tops_current": round(AIPU_TOPS_MAX * load / 100.0, 1),
        "tops_max": AIPU_TOPS_MAX,
        "per_core": per_core,
        "estimated": True,
        "device_fps": metrics.get("device_fps", {}).get("value"),
        "end_to_end_fps": metrics.get("end_to_end_fps", {}).get("value"),
    }
    out["voyager_metrics"] = metrics
    return out


def ensure_self_signed_cert(cert_path: str = "data/ssl/cert.pem", key_path: str = "data/ssl/key.pem") -> Tuple[Optional[str], Optional[str]]:
    """Automatically generates self-signed TLS/SSL certificate & private key with SAN IP extensions and valid keyUsage for Chrome/Safari compatibility."""
    cert_path = os.path.abspath(cert_path)
    key_path = os.path.abspath(key_path)
    local_ip = get_local_ip()

    os.makedirs(os.path.dirname(cert_path), exist_ok=True)
    cnf_path = os.path.join(os.path.dirname(cert_path), "openssl.cnf")

    # Always generate/refresh cert config with SAN IP extensions and digitalSignature keyUsage
    try:
        with open(cnf_path, "w") as f:
            f.write(f"""[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no
[req_distinguished_name]
CN = {local_ip}
[v3_req]
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = {local_ip}
""")
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path, "-days", "3650",
            "-nodes", "-config", cnf_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(cert_path) and os.path.exists(key_path):
            return cert_path, key_path
    except Exception:
        pass

    # Attempt 2: Python cryptography library
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress
        import datetime

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, local_ip)])

        san_list = [
            x509.DNSName(u"localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
        ]
        try:
            san_list.append(x509.IPAddress(ipaddress.ip_address(local_ip)))
        except Exception:
            pass

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=3650)
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ), critical=True
        ).add_extension(
            x509.SubjectAlternativeName(san_list), critical=False
        ).sign(key, hashes.SHA256())

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        return cert_path, key_path
    except Exception:
        pass

    return None, None


def _decode_b64_image(b64_str: str) -> Optional[np.ndarray]:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for low-latency MJPEG video streaming."""
    daemon_threads = True


class NativeHTTPHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for MJPEG streaming and JSON API."""

    server_ref = None  # Reference set by WebServer

    def log_message(self, format, *args):
        # Suppress HTTP access logging noise in terminal
        return

    def _send_json(self, status: int, payload: Any):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:
            return {}

    def _dispatch(self, method: str):
        srv = NativeHTTPHandler.server_ref
        if not srv:
            self.send_error(500)
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}

        if method == "GET" and path in ("/", "/index.html"):
            html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>Axelera Metis Web Interface</h1>")
            return

        if method == "GET" and path.startswith("/video_feed"):
            parts = [p for p in path.split("/") if p]
            key = parts[1] if len(parts) > 1 else "grid"
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for chunk in srv.mjpeg_stream(key):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if method == "GET" and path == "/api/snapshot":
            jpeg = srv.snapshot()
            if not jpeg:
                self._send_json(404, {"error": "No frame available"})
                return
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.send_header("Content-Disposition", 'attachment; filename="snapshot.jpg"')
            self.end_headers()
            self.wfile.write(jpeg)
            return

        if path.startswith("/api/"):
            status, payload = srv.handle_api(method, path, query, self._body() if method != "GET" else {})
            self._send_json(status, payload)
            return
        self.send_error(404, "Not Found")

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


class WebServer:
    """Web Application Server providing remote local-network streaming and API telemetry with HTTPS SSL support."""

    def __init__(self, pipeline, host: str = "0.0.0.0", port: int = 8000, use_https: bool = True,
                 cert_file: Optional[str] = "data/ssl/cert.pem", key_file: Optional[str] = "data/ssl/key.pem",
                 auto_updater: Optional[Any] = None):
        self.pipeline = pipeline
        self.auto_updater = auto_updater
        self.host = host
        self.port = port
        self.use_https = use_https
        self.cert_file = cert_file or "data/ssl/cert.pem"
        self.key_file = key_file or "data/ssl/key.pem"
        self.local_ip = get_local_ip()
        self.lock = threading.Lock()

        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        if FLASK_AVAILABLE:
            self.app = Flask(__name__, template_folder=template_dir)
            if FLASK_CORS_AVAILABLE:
                CORS(self.app)
            self._register_flask_routes()
        else:
            self.app = None

    # ── video ────────────────────────────────────────────────────────────────
    def mjpeg_stream(self, key: str = "grid") -> Generator[bytes, None, None]:
        """Multipart MJPEG generator; registers as a viewer so the camera gets rendered."""
        hub = self.pipeline.preview
        hub.acquire(key)
        last_seq = -1
        try:
            while self.pipeline.is_running:
                item = hub.wait_next(key, last_seq, timeout=1.0)
                if item is None:
                    continue
                last_seq, jpeg = item
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        finally:
            hub.release(key)

    def snapshot(self, key: str = "grid") -> Optional[bytes]:
        return self.pipeline.preview.snapshot(key)

    # ── shared API implementation ────────────────────────────────────────────
    def _cfg_path(self) -> str:
        return getattr(self.auto_updater, "config_path", None) or getattr(self.pipeline, "config_path", None) \
            or "config/config.yaml"

    def _status_payload(self) -> Dict[str, Any]:
        p = self.pipeline
        cams = p.get_camera_status()
        stats = p.get_stats()
        live = [c for c in cams if c["enabled"]]
        avg_fps = (sum(c["fps"] for c in live) / len(live)) if live else 0.0
        return {
            "status": "online" if p.is_running else "offline",
            "hardware": get_hardware_backend_name(p),
            "backend": p.runtime.backend,
            "fps": round(avg_fps, 1),
            "total_fps": stats["total_fps"],
            "active_cameras": len(live),
            "cameras": cams,
            "persons_detected": stats["persons"],
            "faces_recognized": stats["faces_recognized"],
            "telemetry": get_system_telemetry(p),
            "capacity": p.get_capacity_plan(),
            "models": p.get_models_status(),
        }

    def _sync_visualizer(self, model_id: str, enabled: bool):
        vis = getattr(self.pipeline, "visualizer", None)
        if not vis:
            return
        attr = {"human_detector": "draw_boxes", "pose_estimator": "draw_pose", "face_recognizer": "draw_faces",
                "license_plate_recognizer": "draw_plates", "zone_crossing": "draw_zones"}.get(model_id)
        if attr:
            setattr(vis, attr, enabled)

    def handle_api(self, method: str, path: str, query: Dict[str, str], body: Dict[str, Any]) -> Tuple[int, Any]:
        """Single implementation of every /api endpoint (used by Flask and the native server)."""
        p = self.pipeline
        try:
            # ── status / models ─────────────────────────────────────────────
            if method == "GET" and path == "/api/status":
                return 200, self._status_payload()
            if method == "GET" and path == "/api/models":
                return 200, {"models": p.get_models_status()}
            if method == "POST" and path == "/api/models/toggle":
                model_id = body.get("model_id")
                enabled = bool(body.get("enabled", True))
                p.set_model_enabled(model_id, enabled, body.get("camera_id", "all"))
                if body.get("camera_id", "all") in (None, "all"):
                    self._sync_visualizer(model_id, enabled)
                return 200, {"success": True, "model_id": model_id, "enabled": enabled,
                             "models": p.get_models_status()}
            if method == "POST" and path == "/api/toggle":
                model_id = {"boxes": "human_detector", "pose": "pose_estimator",
                            "faces": "face_recognizer"}.get(body.get("feature"))
                if model_id:
                    enabled = bool(body.get("enabled", True))
                    p.set_model_enabled(model_id, enabled)
                    self._sync_visualizer(model_id, enabled)
                return 200, {"success": True, "feature": body.get("feature"), "enabled": body.get("enabled", True)}

            # ── capacity / performance profile ──────────────────────────────
            if method == "GET" and path == "/api/capacity":
                return 200, p.get_capacity_plan()
            if method == "POST" and path in ("/api/capacity/profile", "/api/models/configure_detector"):
                plan = p.set_performance_profile(body.get("profile", "auto"))
                return 200, {"success": True, "profile": p.performance_profile, "capacity": plan,
                             "models": p.get_models_status()}

            # ── cameras ─────────────────────────────────────────────────────
            if method == "GET" and path == "/api/cameras":
                return 200, {"cameras": p.get_camera_status(), "capacity": p.get_capacity_plan()}
            if method == "POST" and path == "/api/cameras":
                cam = p.add_camera(body)
                return 200, {"success": True, "camera": cam, "capacity": p.get_capacity_plan()}
            if path.startswith("/api/cameras/"):
                rest = path[len("/api/cameras/"):].split("/")
                cam_id = urllib.parse.unquote(rest[0])
                if method == "DELETE" and len(rest) == 1:
                    ok = p.remove_camera(cam_id)
                    return (200 if ok else 404), {"success": ok, "camera_id": cam_id,
                                                  "capacity": p.get_capacity_plan()}
                if method == "POST" and len(rest) == 2 and rest[1] == "enable":
                    ok = p.set_camera_enabled(cam_id, bool(body.get("enabled", True)))
                    return (200 if ok else 404), {"success": ok, "camera_id": cam_id,
                                                  "capacity": p.get_capacity_plan()}
            if method == "GET" and path == "/api/events":
                return 200, {"events": list(reversed(p.events[-int(query.get("limit", 50)):]))}

            # ── identities ──────────────────────────────────────────────────
            if method == "GET" and path == "/api/identities":
                ids = p.face_db.list_identities()
                return 200, {"identities": ids, "count": len(ids)}
            if method == "DELETE" and path.startswith("/api/identity/"):
                name = urllib.parse.unquote(path.split("/")[-1])
                return 200, {"success": p.face_db.remove_identity(name), "name": name}
            if method == "POST" and path == "/api/enroll_face":
                name = body.get("name")
                img = body.get("_image")
                if img is None and body.get("image_base64"):
                    img = _decode_b64_image(body["image_base64"])
                if img is None and not body.get("use_live_frame"):
                    return 400, {"error": "Provide an image (file / image_base64) or use_live_frame=true"}
                ok, msg = p.enroll_identity(name, image=img, cam_id=body.get("camera_id"))
                return (200 if ok else 400), ({"success": True, "name": name, "message": msg} if ok
                                              else {"error": msg})

            # ── zones ───────────────────────────────────────────────────────
            if method == "GET" and path == "/api/zones":
                zc = getattr(p, "_zone_crossing", None)
                if zc:
                    return 200, {"zones": zc.get_zone_counts(query.get("cam_id")),
                                 "enabled": p.is_model_enabled("zone_crossing")}
                return 200, {"zones": {}, "enabled": False}
            if method == "GET" and path == "/api/zone_events":
                zc = getattr(p, "_zone_crossing", None)
                since = query.get("since_ts")
                try:
                    since = float(since) if since else None
                except (TypeError, ValueError):
                    since = None
                events = zc.get_events(cam_id=query.get("cam_id"), since_ts=since,
                                       limit=int(query.get("limit", 50))) if zc else []
                return 200, {"events": events, "count": len(events)}
            if method == "POST" and path == "/api/zones/config":
                new_zones = body.get("zones", [])
                p.config.setdefault("zone_crossing", {})["zones"] = new_zones
                from src.analytics.zone_crossing import ZoneCrossingDetector
                p._zone_crossing = ZoneCrossingDetector(new_zones)
                self._persist({"zone_crossing": {"zones": new_zones}})
                return 200, {"success": True, "zone_count": len(new_zones)}

            # ── plates ──────────────────────────────────────────────────────
            if method == "GET" and path == "/api/plates":
                events = p.plate_db.get_recent_events(cam_id=query.get("cam_id"), limit=int(query.get("limit", 50)))
                return 200, {"events": events, "count": len(events)}
            if method == "GET" and path == "/api/plates/unique":
                plates = p.plate_db.list_unique_plates()
                return 200, {"plates": plates, "count": len(plates)}
            if method == "DELETE" and path.startswith("/api/plates/"):
                text = urllib.parse.unquote(path.split("/")[-1])
                return 200, {"success": p.plate_db.remove_plate(text), "plate_text": text}

            # ── auto-update ─────────────────────────────────────────────────
            au = self.auto_updater
            if method == "GET" and path == "/api/update/status":
                return 200, (au.get_status() if au else {"enabled": False, "status": "unavailable"})
            if method == "POST" and path == "/api/update/config":
                if au:
                    au.update_config(body)
                return 200, {"success": True, "status": au.get_status() if au else {"enabled": False, "status": "unavailable"}}
            if method == "POST" and path == "/api/update/check":
                return 200, (au.check_for_updates() if au else {"success": False, "error": "Auto-updater not initialized"})
            if method == "POST" and path == "/api/update/install":
                return 200, (au.apply_update_and_restart() if au else {"success": False, "error": "Auto-updater not initialized"})
        except ValueError as e:
            return 400, {"error": str(e)}
        except Exception as e:
            return 500, {"error": str(e)}
        return 404, {"error": f"Unknown endpoint {method} {path}"}

    def _persist(self, patch: Dict[str, Any]):
        try:
            import yaml
            cfg_path = self._cfg_path()
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    disk_cfg = yaml.safe_load(f) or {}
                for k, v in patch.items():
                    if isinstance(v, dict):
                        disk_cfg.setdefault(k, {}).update(v)
                    else:
                        disk_cfg[k] = v
                with open(cfg_path, "w") as f:
                    yaml.safe_dump(disk_cfg, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"[CONFIG SAVE NOTICE] Failed to update config.yaml: {e}")

    # ── Flask ────────────────────────────────────────────────────────────────
    def _register_flask_routes(self):
        """Registers Flask web routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.mjpeg_stream("grid"), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/video_feed/<cam_id>')
        def video_feed_cam(cam_id):
            return Response(self.mjpeg_stream(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/api/snapshot')
        def api_snapshot():
            jpeg = self.snapshot()
            if not jpeg:
                return jsonify({"error": "No frame available"}), 404
            return Response(jpeg, mimetype='image/jpeg',
                            headers={"Content-Disposition": f'attachment; filename="snapshot_{int(time.time())}.jpg"'})

        @self.app.route('/api/<path:sub>', methods=['GET', 'POST', 'DELETE'])
        def api_dispatch(sub):
            body: Dict[str, Any] = {}
            if request.method != 'GET':
                if request.content_type and 'multipart/form-data' in request.content_type:
                    body = dict(request.form)
                    f = request.files.get('file')
                    if f is not None:
                        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                        if img is not None and img.size:
                            body['_image'] = img
                else:
                    body = request.get_json(silent=True) or {}
            status, payload = self.handle_api(request.method, f"/api/{sub}", request.args.to_dict(), body)
            return jsonify(payload), status

    def start_background(self):
        """Starts web server in a daemon background thread."""
        protocol = "https" if self.use_https else "http"
        print("==========================================================")
        print(f"  [WEB DASHBOARD] Axelera Metis Web Interface Active ({protocol.upper()})")
        print(f"  --> Local Access:   {protocol}://localhost:{self.port}")
        print(f"  --> Network Access: {protocol}://{self.local_ip}:{self.port}")
        if self.use_https:
            print(f"  (Note: Accept self-signed SSL certificate prompt in browser)")
        print("==========================================================")

        cert_p, key_p = None, None
        if self.use_https:
            cert_p, key_p = ensure_self_signed_cert(self.cert_file, self.key_file)

        def run_server():
            if FLASK_AVAILABLE and self.app:
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                try:
                    ssl_arg = (cert_p, key_p) if (cert_p and key_p) else ('adhoc' if self.use_https else None)
                    self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True, ssl_context=ssl_arg)
                    return
                except Exception as e:
                    print(f"[WEB SERVER] Flask start error ({e}). Switching to native HTTP server...")

            # Fallback native HTTP server with SSL
            NativeHTTPHandler.server_ref = self
            server = ThreadingSimpleServer((self.host, self.port), NativeHTTPHandler)
            if self.use_https and cert_p and key_p:
                ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_ctx.load_cert_chain(certfile=cert_p, keyfile=key_p)
                server.socket = ssl_ctx.wrap_socket(server.socket, server_side=True)

            server.serve_forever()

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
