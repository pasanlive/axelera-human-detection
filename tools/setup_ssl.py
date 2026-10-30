#!/usr/bin/env python3
"""
SSL / TLS Certificate Setup Helper for Axelera Metis Web Interface.

Supports:
  1. mkcert (Local CA trusted by browsers for LAN IP 192.168.0.117)
  2. DuckDNS + Let's Encrypt (Free public domain with trusted Let's Encrypt cert)
  3. Import custom existing certificate (.pem / .crt + .key)
  4. Regenerate self-signed certificate with IP SAN extensions
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def get_local_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


def copy_cert_files(src_cert: str, src_key: str, dest_dir: str = "data/ssl") -> None:
    os.makedirs(dest_dir, exist_ok=True)
    dst_cert = os.path.join(dest_dir, "cert.pem")
    dst_key = os.path.join(dest_dir, "key.pem")
    shutil.copyfile(src_cert, dst_cert)
    shutil.copyfile(src_key, dst_key)
    os.chmod(dst_key, 0o600)
    print(f"\n[OK] SSL certificates installed:")
    print(f"  Certificate: {dst_cert}")
    print(f"  Private Key: {dst_key}")
    print("\nRestart the application with HTTPS enabled:")
    print("  python app.py --headless")


def setup_mkcert(ip: str) -> None:
    print(f"\n--- Setting up trusted local SSL with mkcert for IP: {ip} ---")
    if not shutil.which("mkcert"):
        print("[!] 'mkcert' is not installed.")
        print("    To install on Ubuntu/Debian:")
        print("      sudo apt update && sudo apt install -y libnss3-tools")
        print("      curl -JLO \"https://dl.filippo.io/mkcert/latest?for=linux/arm64\"")
        print("      chmod +x mkcert-v*-linux-arm64 && sudo mv mkcert-v*-linux-arm64 /usr/local/bin/mkcert")
        print("\n    To install on macOS:")
        print("      brew install mkcert")
        return

    # Install local CA if not present
    subprocess.run(["mkcert", "-install"], check=True)

    dest_dir = "data/ssl"
    os.makedirs(dest_dir, exist_ok=True)
    cert_path = os.path.join(dest_dir, "cert.pem")
    key_path = os.path.join(dest_dir, "key.pem")

    cmd = [
        "mkcert",
        "-cert-file", cert_path,
        "-key-file", key_path,
        ip, "localhost", "127.0.0.1"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    os.chmod(key_path, 0o600)

    # Show Root CA location
    res = subprocess.run(["mkcert", "-CAROOT"], capture_output=True, text=True)
    caroot = res.stdout.strip()
    root_ca_file = os.path.join(caroot, "rootCA.pem")

    print("\n" + "=" * 65)
    print("  [SUCCESS] Trusted local SSL certificate generated!")
    print(f"  Certificate: {cert_path}")
    print(f"  Private Key: {key_path}")
    print("=" * 65)
    print("\nTo trust this certificate on client computers/phones:")
    print(f"  1. Copy root CA file: {root_ca_file}")
    print("  2. Install it in your browser/OS trusted root store (Keychain on Mac, Cert Manager on Windows).")
    print("  3. Access with green lock: https://" + ip + ":8000")


def setup_duckdns(subdomain: str, token: str) -> None:
    domain = f"{subdomain}.duckdns.org"
    print(f"\n--- Setting up Let's Encrypt for {domain} via DuckDNS DNS-01 ---")
    if not shutil.which("certbot"):
        print("[!] 'certbot' is not installed.")
        print("    Install on Ubuntu: sudo apt update && sudo apt install -y certbot")
        return

    # Verify DuckDNS API
    ip = get_local_ip()
    import urllib.request
    update_url = f"https://www.duckdns.org/update?domains={subdomain}&token={token}&ip={ip}"
    try:
        req = urllib.request.urlopen(update_url, timeout=10)
        resp = req.read().decode('utf-8').strip()
        if resp != "OK":
            print(f"[!] DuckDNS update returned: {resp}. Check your token and subdomain.")
            return
        print(f"[OK] DuckDNS domain {domain} pointed to local IP: {ip}")
    except Exception as e:
        print(f"[!] Failed to update DuckDNS: {e}")
        return

    print("\nObtaining Let's Encrypt wildcard/domain certificate via DNS challenge...")
    print("Run this certbot command on your host:")
    print(f"  sudo certbot certonly --manual --preferred-challenges dns -d {domain}")
    print(f"\nAfter certbot finishes, install the certs with:")
    print(f"  python tools/setup_ssl.py --cert /etc/letsencrypt/live/{domain}/fullchain.pem --key /etc/letsencrypt/live/{domain}/privkey.pem")


def main():
    parser = argparse.ArgumentParser(description="SSL / TLS setup for Axelera Metis Web Interface")
    parser.add_argument("--mkcert", action="store_true", help="Generate browser-trusted certificate using mkcert for local IP")
    parser.add_argument("--ip", default=None, help="IP address to certify (defaults to detected LAN IP)")
    parser.add_argument("--cert", help="Path to existing fullchain certificate (.pem / .crt)")
    parser.add_argument("--key", help="Path to existing private key (.pem / .key)")
    parser.add_argument("--duckdns-domain", help="DuckDNS subdomain (e.g. 'upside-cam')")
    parser.add_argument("--duckdns-token", help="DuckDNS account token")

    args = parser.parse_args()
    local_ip = args.ip or get_local_ip()

    if args.cert and args.key:
        copy_cert_files(args.cert, args.key)
    elif args.mkcert:
        setup_mkcert(local_ip)
    elif args.duckdns_domain and args.duckdns_token:
        setup_duckdns(args.duckdns_domain, args.duckdns_token)
    else:
        print("==========================================================")
        print("  Axelera Metis SSL Configuration Helper")
        print("==========================================================")
        print(f"Local IP detected: {local_ip}")
        print("\nAvailable Options for Free, Valid SSL Certificates:\n")
        print("1. mkcert (Recommended for LAN IP with 100% Green Lock):")
        print(f"   python tools/setup_ssl.py --mkcert --ip {local_ip}")
        print("\n2. Free DuckDNS domain + Let's Encrypt (Trusted everywhere):")
        print("   python tools/setup_ssl.py --duckdns-domain <name> --duckdns-token <token>")
        print("\n3. Import existing certificate & key:")
        print("   python tools/setup_ssl.py --cert /path/to/cert.pem --key /path/to/key.pem")
        print("\n4. Default self-signed certificate (No setup needed, accept warning in browser):")
        print("   python app.py --headless")


if __name__ == "__main__":
    main()
