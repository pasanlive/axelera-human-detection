"""
Camera and RTSP Network Discovery Utility.

Provides:
- ONVIF WS-Discovery (multicast UDP probe to 239.255.255.250:3702)
- Fast multithreaded RTSP/TCP port scanning across local subnets (port 554, 8554)
- RTSP OPTIONS probe to identify server models (Hikvision, Dahua, Uniview, Axis, etc.)
- Stream connection verification with live snapshot preview generation
"""

from __future__ import annotations

import base64
import concurrent.futures
import ipaddress
import os
import re
import socket
import time
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2


WS_DISCOVERY_PROBE = """<?xml version="1.0" encoding="utf-8"?>
<Envelope xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns="http://www.w3.org/2003/05/soap-envelope">
  <Header>
    <wsa:MessageID xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">uuid:{msg_id}</wsa:MessageID>
    <wsa:To xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>
    <wsa:Action xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>
  </Header>
  <Body>
    <Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">
      <Types>dn:NetworkVideoTransmitter tds:Device</Types>
    </Probe>
  </Body>
</Envelope>"""


COMMON_RTSP_PATHS = [
    # Hikvision & compatibles
    "/h264/ch1/main/av_stream",
    "/h264/ch1/sub/av_stream",
    "/live/ch0",
    # Dahua / Amcrest
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
    # Uniview / Tiandy
    "/unicast/c1/s0/live",
    "/media/video1",
    # Generic ONVIF / RTSP
    "/onvif1",
    "/stream1",
    "/stream2",
    "/ch01.264",
    "/video",
    "",
]


def get_local_ip() -> str:
    """Detects active primary IPv4 address of the host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually transmit packets, just routes to local interface
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def get_default_subnet() -> str:
    """Returns /24 subnet for local host, e.g. '192.168.0.0/24'."""
    ip = get_local_ip()
    if ip.startswith("127."):
        return "192.168.1.0/24"
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    return "192.168.0.0/24"


def discover_onvif_cameras(timeout_s: float = 1.8) -> List[Dict[str, Any]]:
    """Broadcasts ONVIF WS-Discovery probe and gathers device responses."""
    devices: List[Dict[str, Any]] = []
    seen_ips: Set[str] = set()

    multicast_group = "239.255.255.250"
    port = 3702
    msg = WS_DISCOVERY_PROBE.format(msg_id=str(uuid.uuid4())).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
    sock.settimeout(timeout_s)

    try:
        sock.sendto(msg, (multicast_group, port))
        t_start = time.time()

        while time.time() - t_start < timeout_s:
            try:
                data, addr = sock.recvfrom(65535)
                ip = addr[0]
                if ip in seen_ips:
                    continue

                text = data.decode("utf-8", errors="ignore")
                dev_info = _parse_onvif_probe_match(text, ip)
                seen_ips.add(ip)
                devices.append(dev_info)
            except socket.timeout:
                break
            except Exception:
                continue
    except Exception:
        pass
    finally:
        sock.close()

    return devices


def _parse_onvif_probe_match(xml_text: str, default_ip: str) -> Dict[str, Any]:
    """Extracts hardware metadata, XAddrs, and vendor from ONVIF probe response."""
    info: Dict[str, Any] = {
        "ip": default_ip,
        "port": 554,
        "protocol": "onvif",
        "vendor": "Generic ONVIF",
        "model": "Network Camera",
        "name": f"ONVIF Camera ({default_ip})",
        "xaddrs": "",
        "uuid": "",
    }

    try:
        # Fast regex extraction to avoid XML namespace mismatch issues
        xaddrs_m = re.search(r"<(?:[^:>]+:)?XAddrs>([^<]+)</(?:[^:>]+:)?XAddrs>", xml_text)
        if xaddrs_m:
            xaddrs = xaddrs_m.group(1).strip()
            info["xaddrs"] = xaddrs
            parsed = urllib.parse.urlparse(xaddrs.split()[0])
            if parsed.hostname:
                info["ip"] = parsed.hostname

        scopes_m = re.search(r"<(?:[^:>]+:)?Scopes>([^<]+)</(?:[^:>]+:)?Scopes>", xml_text)
        if scopes_m:
            scopes = scopes_m.group(1).split()
            for s in scopes:
                s_lower = s.lower()
                if "hardware/" in s_lower:
                    info["model"] = urllib.parse.unquote(s.split("hardware/")[-1])
                elif "name/" in s_lower:
                    info["name"] = urllib.parse.unquote(s.split("name/")[-1]).replace("_", " ")
                elif "manufacturer/" in s_lower:
                    info["vendor"] = urllib.parse.unquote(s.split("manufacturer/")[-1])
                elif "hikvision" in s_lower:
                    info["vendor"] = "Hikvision"
                elif "dahua" in s_lower:
                    info["vendor"] = "Dahua"
                elif "axis" in s_lower:
                    info["vendor"] = "Axis"
                elif "uniview" in s_lower:
                    info["vendor"] = "Uniview"

        uuid_m = re.search(r"<(?:[^:>]+:)?Address>urn:uuid:([^<]+)</(?:[^:>]+:)?Address>", xml_text)
        if uuid_m:
            info["uuid"] = uuid_m.group(1).strip()

    except Exception:
        pass

    return info


def probe_rtsp_service(ip: str, port: int = 554, timeout_s: float = 0.5) -> Optional[Dict[str, Any]]:
    """Probes an IP for RTSP service on given port and inspects Server banner."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)

    try:
        sock.connect((ip, port))
        # Send simple RTSP OPTIONS request
        req = f"OPTIONS rtsp://{ip}:{port}/ RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: UpsideCameraScanner/1.0\r\n\r\n"
        sock.sendall(req.encode("ascii"))
        resp = sock.recv(2048).decode("ascii", errors="ignore")

        if "RTSP/1.0" in resp:
            server_header = ""
            for line in resp.splitlines():
                if line.lower().startswith("server:"):
                    server_header = line.split(":", 1)[1].strip()

            vendor = "Generic RTSP"
            srv_low = server_header.lower()
            if "hikvision" in srv_low or "dvr" in srv_low:
                vendor = "Hikvision"
            elif "dahua" in srv_low:
                vendor = "Dahua"
            elif "axis" in srv_low:
                vendor = "Axis"
            elif "live555" in srv_low:
                vendor = "Live555 Streamer"

            return {
                "ip": ip,
                "port": port,
                "protocol": "rtsp",
                "vendor": vendor,
                "server_banner": server_header,
                "name": f"{vendor} Camera ({ip})",
            }
        else:
            # Port was open, treat as potential RTSP endpoint
            return {
                "ip": ip,
                "port": port,
                "protocol": "rtsp",
                "vendor": "RTSP Device",
                "name": f"Camera ({ip}:{port})",
            }
    except Exception:
        return None
    finally:
        sock.close()


def scan_subnet_rtsp(subnet: str, timeout_s: float = 0.4, max_workers: int = 64) -> List[Dict[str, Any]]:
    """Scans all hosts in a /24 subnet for open RTSP port 554/8554."""
    try:
        network = ipaddress.ip_network(subnet, strict=False)
    except Exception:
        network = ipaddress.ip_network("192.168.0.0/24")

    hosts = [str(h) for h in network.hosts()]
    # Cap scan size to 256 hosts for fast responsive completion
    if len(hosts) > 256:
        hosts = hosts[:256]

    results: List[Dict[str, Any]] = []

    def _check_host(ip: str) -> List[Dict[str, Any]]:
        devs = []
        for port in (554, 8554):
            dev = probe_rtsp_service(ip, port=port, timeout_s=timeout_s)
            if dev:
                devs.append(dev)
        return devs

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_host, h): h for h in hosts}
        for future in concurrent.futures.as_completed(futures):
            try:
                found = future.result()
                if found:
                    results.extend(found)
            except Exception:
                pass

    return results


def discover_network_cameras(subnet: Optional[str] = None,
                             existing_cameras: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Combines ONVIF discovery and subnet RTSP port scan to discover cameras.

    Flags whether any discovered device is already configured in the system.
    """
    target_subnet = subnet or get_default_subnet()
    configured_ips: Dict[str, str] = {}  # ip -> cam_id

    for cam in (existing_cameras or []):
        src = cam.get("source", "")
        cid = cam.get("id", "cam")
        parsed = urllib.parse.urlparse(src)
        if parsed.hostname:
            configured_ips[parsed.hostname] = cid

    merged_devices: Dict[str, Dict[str, Any]] = {}

    # 1. Run ONVIF WS-Discovery probe
    try:
        onvif_devs = discover_onvif_cameras(timeout_s=1.8)
        for dev in onvif_devs:
            ip = dev["ip"]
            merged_devices[ip] = dev
    except Exception:
        pass

    # 2. Run Subnet Port Scan (catches cameras with ONVIF disabled)
    try:
        rtsp_devs = scan_subnet_rtsp(target_subnet, timeout_s=0.4, max_workers=64)
        for dev in rtsp_devs:
            ip = dev["ip"]
            if ip in merged_devices:
                # Merge server banner into existing onvif record
                if dev.get("server_banner"):
                    merged_devices[ip]["server_banner"] = dev["server_banner"]
            else:
                merged_devices[ip] = dev
    except Exception:
        pass

    device_list: List[Dict[str, Any]] = []
    for ip, dev in sorted(merged_devices.items()):
        is_configured = ip in configured_ips
        dev["already_configured"] = is_configured
        dev["configured_id"] = configured_ips.get(ip)
        dev["suggested_paths"] = COMMON_RTSP_PATHS

        # Guess default main path based on vendor
        v_low = dev.get("vendor", "").lower()
        if "hikvision" in v_low:
            dev["default_path"] = "/h264/ch1/main/av_stream"
            dev["default_sub_path"] = "/h264/ch1/sub/av_stream"
        elif "dahua" in v_low or "amcrest" in v_low:
            dev["default_path"] = "/cam/realmonitor?channel=1&subtype=0"
            dev["default_sub_path"] = "/cam/realmonitor?channel=1&subtype=1"
        elif "uniview" in v_low:
            dev["default_path"] = "/unicast/c1/s0/live"
            dev["default_sub_path"] = "/live/ch0"
        else:
            dev["default_path"] = "/h264/ch1/main/av_stream"
            dev["default_sub_path"] = "/live/ch0"

        device_list.append(dev)

    return {
        "success": True,
        "subnet": target_subnet,
        "local_ip": get_local_ip(),
        "count": len(device_list),
        "cameras": device_list,
    }


def build_rtsp_url(source: str, username: Optional[str] = None,
                   password: Optional[str] = None) -> str:
    """Builds a formatted RTSP URL with URL-encoded credentials."""
    s = source.strip()
    if not s.startswith("rtsp://") and not s.startswith("rtsps://"):
        s = f"rtsp://{s}"

    parsed = urllib.parse.urlsplit(s)
    user = username if username is not None else parsed.username
    pwd = password if password is not None else parsed.password

    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    if user or pwd:
        user_enc = urllib.parse.quote(user or "", safe="")
        pwd_enc = urllib.parse.quote(pwd or "", safe="")
        netloc = f"{user_enc}:{pwd_enc}@{netloc}"

    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def test_camera_stream(source: str, username: Optional[str] = None,
                       password: Optional[str] = None,
                       timeout_s: float = 6.0) -> Dict[str, Any]:
    """Tests opening an RTSP camera stream and capturing a preview frame."""
    url = build_rtsp_url(source, username=username, password=password)
    # Mask password for safe logging
    safe_url = re.sub(r":([^@]+)@", ":***@", url)

    # Force TCP transport in OpenCV FFMPEG backend for reliability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_s * 1000))

    t0 = time.time()
    try:
        if not cap.isOpened():
            return {
                "success": False,
                "url": safe_url,
                "error": f"Failed to connect to RTSP camera stream within {timeout_s:.1f}s. Check IP, credentials, and stream path.",
            }

        ret, frame = cap.read()
        latency_ms = round((time.time() - t0) * 1000.0, 1)

        if not ret or frame is None or frame.size == 0:
            return {
                "success": False,
                "url": safe_url,
                "error": "Connected to camera RTSP port, but received an empty or corrupted video frame.",
            }

        h, w = frame.shape[:2]
        fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 15.0), 1)

        # Scale down preview to max 640px width for fast responsive JSON payload
        preview_frame = frame
        if w > 640:
            new_w = 640
            new_h = int(h * (640 / w))
            preview_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ret_enc, buf = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret_enc:
            return {
                "success": False,
                "url": safe_url,
                "error": "Failed to encode snapshot preview image.",
            }

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        preview_uri = f"data:image/jpeg;base64,{b64}"

        return {
            "success": True,
            "url": safe_url,
            "raw_url": url,
            "width": w,
            "height": h,
            "fps": fps,
            "latency_ms": latency_ms,
            "preview_image": preview_uri,
            "message": f"Stream verified: {w}x{h} @ {fps} FPS ({latency_ms}ms connect latency)",
        }
    except Exception as e:
        return {
            "success": False,
            "url": safe_url,
            "error": f"Connection exception: {str(e)}",
        }
    finally:
        try:
            cap.release()
        except Exception:
            pass
