#!/usr/bin/env python3
"""
Camera and RTSP Network Discovery Tool.

Discovers ONVIF and RTSP cameras across local subnets using:
1. ONVIF WS-Discovery (multicast UDP probe to 239.255.255.250:3702).
2. Fast multithreaded RTSP/TCP port scanning (ports 554, 8554).
3. RTSP OPTIONS banner inspection (Hikvision, Dahua, Uniview, Axis, etc.).
4. Optional stream connection verification and snapshot preview capturing.
5. Cross-referencing against currently configured cameras in config.yaml.

Usage:
  python3 tools/discover_cameras.py
  python3 tools/discover_cameras.py --subnet 192.168.1.0/24
  python3 tools/discover_cameras.py --all-interfaces
  python3 tools/discover_cameras.py --test --user admin --password secret
  python3 tools/discover_cameras.py --test-url rtsp://admin:pass@192.168.0.102:554/live/ch0
  python3 tools/discover_cameras.py --json
"""

import argparse
import base64
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.camera_discovery import (
    COMMON_RTSP_PATHS,
    build_rtsp_url,
    discover_network_cameras,
    discover_onvif_cameras,
    get_default_subnet,
    get_local_ip,
    scan_subnet_rtsp,
    test_camera_stream,
)


def get_all_local_subnets() -> List[str]:
    """Finds all non-loopback IPv4 subnets across local network interfaces."""
    subnets = set()
    try:
        import netifaces  # type: ignore

        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr.get("addr")
                    netmask = addr.get("netmask")
                    if ip and not ip.startswith("127.") and netmask:
                        try:
                            import ipaddress
                            net = ipaddress.ip_network(f"{ip}/{netmask}", strict=False)
                            subnets.add(str(net))
                        except Exception:
                            pass
    except ImportError:
        pass

    if not subnets:
        primary = get_default_subnet()
        if primary:
            subnets.add(primary)

    return sorted(list(subnets))


def load_configured_cameras() -> List[Dict[str, Any]]:
    """Loads existing cameras from system or project config.yaml."""
    paths_to_check = [
        Path.home() / ".upside-ai" / "config" / "config.yaml",
        PROJECT_ROOT / "config" / "config.yaml",
    ]
    for p in paths_to_check:
        if p.exists():
            try:
                import yaml  # type: ignore
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cams = cfg.get("cameras", [])
                if isinstance(cams, list):
                    return cams
            except Exception:
                pass
    return []


def print_banner() -> None:
    print("=" * 72)
    print("  Upside AI - Network Camera & RTSP Discovery Tool")
    print("=" * 72)


def format_table(devices: List[Dict[str, Any]]) -> str:
    """Formats discovered camera list into a clean terminal table."""
    if not devices:
        return "  No network cameras or RTSP services detected."

    lines = []
    header = f" {'#':<3} {'IP Address':<16} {'Port':<6} {'Vendor':<14} {'Protocol':<8} {'Status':<16} {'Name/Model'}"
    lines.append(header)
    lines.append("-" * len(header) + "-" * 15)

    for idx, dev in enumerate(devices, 1):
        ip = dev.get("ip", "unknown")
        port = str(dev.get("port", 554))
        vendor = dev.get("vendor", "Unknown")[:13]
        proto = dev.get("protocol", "rtsp").upper()
        if dev.get("already_configured"):
            status = f"[CONFIGURED:{dev.get('configured_id', 'cam')}]"
        else:
            status = "[AVAILABLE]"
        name = dev.get("name") or dev.get("model") or "Camera"

        lines.append(f" {idx:<3} {ip:<16} {port:<6} {vendor:<14} {proto:<8} {status:<16} {name}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover available ONVIF and RTSP cameras across local subnets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subnet",
        "-s",
        nargs="+",
        help="Subnet CIDR(s) to scan (e.g. 192.168.0.0/24 10.0.0.0/24). Defaults to local subnet.",
    )
    parser.add_argument(
        "--all-interfaces",
        "-a",
        action="store_true",
        help="Scan all active IPv4 interfaces and subnets.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.4,
        help="Timeout in seconds for RTSP port probing (default: 0.4s).",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Test RTSP stream connections on discovered cameras and capture snapshot preview.",
    )
    parser.add_argument(
        "--user",
        "-u",
        default="",
        help="Username for RTSP authentication test (e.g. admin).",
    )
    parser.add_argument(
        "--password",
        "-p",
        default="",
        help="Password for RTSP authentication test.",
    )
    parser.add_argument(
        "--test-url",
        help="Directly test an RTSP stream URL and output resolution, FPS, and latency.",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "tcp", "udp"],
        default="auto",
        help="RTSP transport protocol to test (default: auto with TCP->UDP fallback).",
    )
    parser.add_argument(
        "--save-preview",
        metavar="DIR",
        help="Save captured preview JPEG images to the specified directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format for scripting or API integration.",
    )

    args = parser.parse_args()

    # Case 1: Direct single URL test
    if args.test_url:
        print_banner()
        print(f"Testing RTSP stream URL: {args.test_url}")
        print(f"Transport mode: {args.transport.upper()}")
        print("-" * 72)
        res = test_camera_stream(
            args.test_url,
            username=args.user or None,
            password=args.password or None,
            transport=args.transport,
            timeout_s=6.0,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0 if res.get("success") else 1

        if res.get("success"):
            print(f"  [SUCCESS] {res.get('message')}")
            print(f"  Resolution: {res.get('width')}x{res.get('height')}")
            print(f"  Framerate:  {res.get('fps')} FPS")
            print(f"  Latency:    {res.get('latency_ms')} ms")
            print(f"  Transport:  {str(res.get('transport')).upper()}")
            if args.save_preview and res.get("preview_image"):
                out_dir = Path(args.save_preview)
                out_dir.mkdir(parents=True, exist_ok=True)
                img_path = out_dir / "preview_test.jpg"
                b64_data = res["preview_image"].split(",", 1)[-1]
                img_path.write_bytes(base64.b64decode(b64_data))
                print(f"  Preview:    Saved snapshot to {img_path}")
            return 0
        else:
            print(f"  [FAILED] {res.get('error')}")
            return 1

    # Case 2: Full network discovery
    existing_cams = load_configured_cameras()

    if args.all_interfaces:
        subnets_to_scan = get_all_local_subnets()
    elif args.subnet:
        subnets_to_scan = args.subnet
    else:
        subnets_to_scan = [get_default_subnet()]

    if not args.json:
        print_banner()
        print(f"Host Local IP:      {get_local_ip()}")
        print(f"Subnets to scan:    {', '.join(subnets_to_scan)}")
        print(f"Configured cameras: {len(existing_cams)} loaded from config.yaml")
        print("\nScanning network (ONVIF WS-Discovery + Multithreaded RTSP probe)...")
        t0 = time.time()

    all_devices: Dict[str, Dict[str, Any]] = {}

    # Scan each subnet
    for sub in subnets_to_scan:
        res = discover_network_cameras(subnet=sub, existing_cameras=existing_cams)
        for dev in res.get("cameras", []):
            ip = dev["ip"]
            if ip not in all_devices:
                all_devices[ip] = dev
            else:
                # Merge fields
                if not all_devices[ip].get("server_banner") and dev.get("server_banner"):
                    all_devices[ip]["server_banner"] = dev["server_banner"]

    devices_list = list(all_devices.values())

    # Optional connection verification
    if args.test and devices_list:
        if not args.json:
            print(f"\nTesting stream connections for {len(devices_list)} discovered camera(s)...")

        for dev in devices_list:
            ip = dev["ip"]
            port = dev.get("port", 554)
            path = dev.get("default_path", "/h264/ch1/main/av_stream")
            url = f"rtsp://{ip}:{port}{path}"

            test_res = test_camera_stream(
                url,
                username=args.user or None,
                password=args.password or None,
                transport=args.transport,
                timeout_s=3.0,
            )
            dev["stream_test"] = test_res

            if args.save_preview and test_res.get("success") and test_res.get("preview_image"):
                out_dir = Path(args.save_preview)
                out_dir.mkdir(parents=True, exist_ok=True)
                clean_ip = ip.replace(".", "_")
                img_path = out_dir / f"cam_{clean_ip}.jpg"
                b64_data = test_res["preview_image"].split(",", 1)[-1]
                img_path.write_bytes(base64.b64decode(b64_data))
                dev["preview_file"] = str(img_path)

    if args.json:
        out_payload = {
            "success": True,
            "host_ip": get_local_ip(),
            "subnets": subnets_to_scan,
            "count": len(devices_list),
            "cameras": devices_list,
        }
        print(json.dumps(out_payload, indent=2))
        return 0

    elapsed = time.time() - t0
    print(f"Scan completed in {elapsed:.2f}s.\n")
    print(format_table(devices_list))

    # Print suggested RTSP paths for each available camera
    available = [d for d in devices_list if not d.get("already_configured")]
    if available:
        print("\n" + "=" * 72)
        print("  Suggested RTSP Stream Configurations for New Cameras:")
        print("=" * 72)
        for dev in available:
            ip = dev.get("ip")
            vendor = dev.get("vendor", "Generic")
            name = dev.get("name") or dev.get("model") or "Camera"
            main_path = dev.get("default_path", "/h264/ch1/main/av_stream")
            sub_path = dev.get("default_sub_path", "/live/ch0")
            print(f"\n- Camera: {name} ({vendor} at {ip})")
            print(f"  Main Stream: rtsp://<username>:<password>@{ip}:554{main_path}")
            print(f"  Sub Stream:  rtsp://<username>:<password>@{ip}:554{sub_path}")

            if "stream_test" in dev:
                t = dev["stream_test"]
                if t.get("success"):
                    print(f"  [STREAM TEST] Verified: {t.get('width')}x{t.get('height')} @ {t.get('fps')} FPS via {str(t.get('transport')).upper()}")
                else:
                    print(f"  [STREAM TEST] Failed: {t.get('error')}")

    print("\nTip: To test a specific stream with credentials, run:")
    print("  python3 tools/discover_cameras.py --test-url rtsp://<user>:<pass>@<ip>:554/<path>")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
