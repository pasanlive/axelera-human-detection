"""
WebServer: Low-Latency MJPEG Web Streaming and REST API for Axelera Metis System.
Supports Flask and native Python http.server fallback for zero-dependency remote local network access.
"""

import os
import sys
import time
import json
import socket
import threading
from typing import Dict, Any, Generator
import cv2
import numpy as np

# Optional Flask support
try:
    from flask import Flask, render_template, Response, jsonify, request, send_file
    try:
        from flask_cors import CORS
        FLASK_CORS_AVAILABLE = True
    except ImportError:
        FLASK_CORS_AVAILABLE = False
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Fallback Python standard library HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn


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


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for low-latency MJPEG video streaming."""
    daemon_threads = True


class NativeHTTPHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for MJPEG streaming and JSON API."""
    
    server_ref = None  # Reference set by WebServer

    def log_message(self, format, *args):
        # Suppress HTTP access logging noise in terminal
        return

    def do_GET(self):
        srv = NativeHTTPHandler.server_ref
        if not srv:
            self.send_error(500)
            return

        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
            if os.path.exists(html_path):
                with open(html_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>Axelera Metis Web Interface</h1>")

        elif self.path.startswith('/video_feed'):
            parts = self.path.split('/')
            cam_id = parts[-1] if len(parts) > 2 and parts[-1] != 'video_feed' else 'grid'
            
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while srv.pipeline.is_running:
                frame_bytes = None
                with srv.lock:
                    if cam_id == 'grid':
                        frame_bytes = srv.latest_grid_jpeg
                    else:
                        frame_bytes = srv.latest_cam_jpegs.get(cam_id)

                if frame_bytes is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(1.0 / 30.0)

        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            cam_list = []
            if hasattr(srv.pipeline, 'stream_manager'):
                for cid, cam in srv.pipeline.stream_manager.cameras.items():
                    is_open = getattr(cam, 'is_running', False)
                    if hasattr(cam, 'cap') and cam.cap is not None:
                        is_open = cam.cap.isOpened()
                    cam_list.append({
                        "id": cid,
                        "name": cam.name,
                        "fps": round(getattr(cam, 'fps', 0.0), 1),
                        "is_opened": is_open
                    })

            payload = {
                "status": "online" if srv.pipeline.is_running else "offline",
                "hardware": "Axelera Metis AIPU (4 Cores)",
                "fps": 30.0,
                "active_cameras": len(cam_list),
                "cameras": cam_list,
                "persons_detected": srv.total_persons,
                "faces_recognized": srv.total_faces
            }
            self.wfile.write(json.dumps(payload).encode('utf-8'))

        elif self.path == '/api/snapshot':
            with srv.lock:
                jpeg_data = srv.latest_grid_jpeg
            if jpeg_data:
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-Disposition', 'attachment; filename="snapshot.jpg"')
                self.end_headers()
                self.wfile.write(jpeg_data)
            else:
                self.send_error(404, "No frame available")

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        srv = NativeHTTPHandler.server_ref
        if self.path == '/api/toggle' and srv:
            length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode('utf-8'))
                feature = data.get('feature')
                enabled = data.get('enabled', True)
                vis = getattr(srv.pipeline, 'visualizer', None)
                if vis:
                    if feature == 'boxes': vis.draw_boxes = enabled
                    elif feature == 'pose': vis.draw_pose = enabled
                    elif feature == 'faces': vis.draw_faces = enabled

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_error(400, str(e))
        else:
            self.send_error(404)


class WebServer:
    """Web Application Server providing remote local-network streaming and API telemetry."""

    def __init__(self, pipeline, host: str = "0.0.0.0", port: int = 8000):
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.local_ip = get_local_ip()

        self.latest_grid_jpeg = None
        self.latest_cam_jpegs: Dict[str, bytes] = {}
        self.lock = threading.Lock()

        self.total_persons = 0
        self.total_faces = 0

        template_dir = os.path.join(os.path.dirname(__file__), "templates")

        if FLASK_AVAILABLE:
            self.app = Flask(__name__, template_folder=template_dir)
            if FLASK_CORS_AVAILABLE:
                CORS(self.app)
            self._register_flask_routes()
        else:
            self.app = None

    def update_frame(self, grid_frame: np.ndarray, individual_frames: Dict[str, np.ndarray]):
        """Thread-safe update of latest rendered frames into JPEG buffers for streaming."""
        try:
            if grid_frame is not None and grid_frame.size > 0:
                ret, jpeg = cv2.imencode('.jpg', grid_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    with self.lock:
                        self.latest_grid_jpeg = jpeg.tobytes()

            if individual_frames:
                for cam_id, frame in individual_frames.items():
                    if frame is not None and frame.size > 0:
                        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ret:
                            with self.lock:
                                self.latest_cam_jpegs[cam_id] = jpeg.tobytes()
        except Exception:
            pass

    def _generate_mjpeg_stream(self, cam_id: str = "grid") -> Generator[bytes, None, None]:
        """Generates HTTP multipart MJPEG stream from latest frame buffer."""
        while self.pipeline.is_running:
            frame_bytes = None
            with self.lock:
                if cam_id == "grid":
                    frame_bytes = self.latest_grid_jpeg
                else:
                    frame_bytes = self.latest_cam_jpegs.get(cam_id)

            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1.0 / 30.0)

    def _register_flask_routes(self):
        """Registers Flask web routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_mjpeg_stream("grid"),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/video_feed/<cam_id>')
        def video_feed_cam(cam_id):
            return Response(self._generate_mjpeg_stream(cam_id),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/api/status')
        def api_status():
            cam_list = []
            if hasattr(self.pipeline, 'stream_manager'):
                for cid, cam in self.pipeline.stream_manager.cameras.items():
                    is_open = getattr(cam, 'is_running', False)
                    if hasattr(cam, 'cap') and cam.cap is not None:
                        is_open = cam.cap.isOpened()
                    cam_list.append({
                        "id": cid,
                        "name": cam.name,
                        "fps": round(getattr(cam, 'fps', 0.0), 1),
                        "is_opened": is_open
                    })

            return jsonify({
                "status": "online" if self.pipeline.is_running else "offline",
                "hardware": "Axelera Metis AIPU (4 Cores)",
                "fps": 30.0,
                "active_cameras": len(cam_list),
                "cameras": cam_list,
                "persons_detected": self.total_persons,
                "faces_recognized": self.total_faces
            })

        @self.app.route('/api/toggle', methods=['POST'])
        def api_toggle():
            data = request.json or {}
            feature = data.get('feature')
            enabled = data.get('enabled', True)

            vis = getattr(self.pipeline, 'visualizer', None)
            if vis:
                if feature == 'boxes': vis.draw_boxes = enabled
                elif feature == 'pose': vis.draw_pose = enabled
                elif feature == 'faces': vis.draw_faces = enabled

            return jsonify({"success": True, "feature": feature, "enabled": enabled})

        @self.app.route('/api/snapshot')
        def api_snapshot():
            with self.lock:
                if self.latest_grid_jpeg:
                    snapshot_filename = f"snapshot_{int(time.time())}.jpg"
                    filepath = os.path.join("data", snapshot_filename)
                    os.makedirs("data", exist_ok=True)
                    with open(filepath, "wb") as f:
                        f.write(self.latest_grid_jpeg)
                    return send_file(filepath, mimetype='image/jpeg', as_attachment=True)
            return jsonify({"error": "No frame available"}), 404

        @self.app.route('/api/identities', methods=['GET'])
        def api_list_identities():
            identities = []
            if hasattr(self.pipeline, 'face_db'):
                identities = self.pipeline.face_db.list_identities()
            return jsonify({"identities": identities, "count": len(identities)})

        @self.app.route('/api/identity/<name>', methods=['DELETE'])
        def api_delete_identity(name):
            success = False
            if hasattr(self.pipeline, 'face_db'):
                success = self.pipeline.face_db.remove_identity(name)
            return jsonify({"success": success, "name": name})

        @self.app.route('/api/enroll_face', methods=['POST'])
        def api_enroll_face():
            import base64
            name = None
            if request.content_type and 'multipart/form-data' in request.content_type:
                name = request.form.get('name')
            elif request.is_json:
                name = request.json.get('name')

            if not name:
                return jsonify({"error": "Identity name is required"}), 400

            img = None
            if 'file' in request.files:
                file = request.files['file']
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            elif request.is_json and 'image_base64' in request.json:
                b64_str = request.json['image_base64']
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_data = base64.b64decode(b64_str)
                file_bytes = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            elif request.is_json and request.json.get('use_live_frame'):
                with self.lock:
                    if self.latest_grid_jpeg:
                        file_bytes = np.frombuffer(self.latest_grid_jpeg, np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None or img.size == 0:
                return jsonify({"error": "No valid face image provided"}), 400

            success = False
            if hasattr(self.pipeline, 'face_recognizer'):
                success = self.pipeline.face_recognizer.enroll_identity(name, img)

            if success:
                return jsonify({"success": True, "name": name, "message": f"Successfully enrolled '{name}'."})
            else:
                return jsonify({"error": "Could not detect or extract face features from image."}), 400

    def start_background(self):
        """Starts web server in a daemon background thread."""
        print("==========================================================")
        print(f"  [WEB DASHBOARD] Axelera Metis Web Interface Active")
        print(f"  --> Local Access:   http://localhost:{self.port}")
        print(f"  --> Network Access: http://{self.local_ip}:{self.port}")
        print("==========================================================")

        def run_server():
            if FLASK_AVAILABLE and self.app:
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                try:
                    self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)
                    return
                except Exception as e:
                    print(f"[WEB SERVER] Flask start error ({e}). Switching to native HTTP server...")

            # Fallback native multi-threaded HTTP server
            NativeHTTPHandler.server_ref = self
            server = ThreadingSimpleServer((self.host, self.port), NativeHTTPHandler)
            server.serve_forever()

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
