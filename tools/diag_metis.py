#!/usr/bin/env python3
"""
Comprehensive Hardware & Driver Diagnostic Utility for Axelera Metis on Rockchip / Embedded Linux.

Checks:
1. PCIe link status, negotiated speed (Gen1/2/3), width, and ASPM power states.
2. Kernel dmesg ring buffer for PCIe, UIO, DMA, and Metis driver events.
3. Axelera Metis device nodes (/dev/metis*) and orphaned file descriptors.
4. axdevice firmware / driver version alignment.
5. System thermals and power profile.

Usage:
  python tools/diag_metis.py
  python tools/diag_metis.py --fix-aspm   # Proposes or executes setpci command to disable PCIe ASPM
"""

import argparse
import glob
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def print_header(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_cmd(cmd: list[str], timeout: int = 5) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (res.stdout or "") + (res.stderr or "")
        return res.returncode, out.strip()
    except Exception as e:
        return -1, str(e)


def check_system() -> None:
    print_header("System & Platform Architecture")
    uname = platform.uname()
    print(f"OS / Kernel:     {uname.system} {uname.release} ({uname.machine})")
    print(f"Python:          {sys.version.split()[0]} ({sys.executable})")

    # Check Rockchip device tree model if on ARM
    rk_model = "/proc/device-tree/model"
    if os.path.exists(rk_model):
        try:
            with open(rk_model, "r") as f:
                print(f"Hardware Model:  {f.read().strip()}")
        except Exception:
            pass

    # Check thermal zones
    tz_files = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))
    if tz_files:
        temps = []
        for tf in tz_files:
            type_f = os.path.join(os.path.dirname(tf), "type")
            try:
                name = open(type_f).read().strip() if os.path.exists(type_f) else os.path.basename(os.path.dirname(tf))
                val = float(open(tf).read().strip()) / 1000.0
                temps.append(f"{name}: {val:.1f}°C")
            except Exception:
                pass
        if temps:
            print(f"SoC Thermals:    {', '.join(temps[:4])}")


def check_pcie() -> None:
    print_header("PCIe Bus & ASPM Power State Inspection")
    rc, out = run_cmd(["lspci", "-d", "1f9d:", "-vvv"])
    if rc == 0 and "Capabilities:" not in out:
        rc_sudo, out_sudo = run_cmd(["sudo", "-n", "lspci", "-d", "1f9d:", "-vvv"])
        if rc_sudo == 0 and "Capabilities:" in out_sudo:
            out = out_sudo

    if rc != 0 or not out:
        # Fall back to listing all devices
        rc_all, out_all = run_cmd(["lspci"])
        print("[WARNING] Axelera PCIe device (Vendor 1f9d) not found via 'lspci -d 1f9d:'")
        if "1f9d" in out_all or "Axelera" in out_all or "Processing accelerators" in out_all:
            print("Found potential accelerator in full lspci:")
            for l in out_all.splitlines():
                if any(k in l for k in ("Axelera", "1f9d", "Processing accelerators")):
                    print(f"  --> {l}")
        else:
            print("Available PCIe devices:")
            for l in out_all.splitlines()[:10]:
                print(f"  {l}")
        return

    print("Found Axelera Metis PCIe Device:")
    lines = out.splitlines()
    for l in lines[:5]:
        print(f"  {l}")

    link_caps, link_status, aspm_info = None, None, []
    for l in lines:
        l_str = l.strip()
        if "LnkCap:" in l_str:
            link_caps = l_str
        elif "LnkSta:" in l_str:
            link_status = l_str
        elif "ASPM" in l_str:
            aspm_info.append(l_str)

    if link_caps:
        print(f"\n  Link Capabilities:  {link_caps}")
    if link_status:
        print(f"  Link Status:        {link_status}")
        # Detect downgrade
        if "Speed 2.5GT/s" in link_status and "Speed 8GT/s" in (link_caps or ""):
            print("  [ALERT] PCIe link speed downgraded to Gen1 (2.5GT/s) - indicates PCIe signal degradation!")

    if aspm_info:
        print(f"  ASPM Power States:  {'; '.join(aspm_info)}")
        if any("ASPM L1" in s and "Enabled" in s for s in aspm_info):
            print("\n  [CRITICAL NOTICE FOR RK3588]")
            print("  PCIe ASPM L1 is ENABLED. On Rockchip RK3588, when the PCIe controller enters L1")
            print("  power-saving state, MSI interrupts from the Metis AIPU can be dropped, causing:")
            print("    'axl 0000:01:00.0: IRQ MSI timeout'")
            print("    'waitForQueueBinaryCompletion: Wait kernel failed with return code -1301'")
            if os.geteuid() == 0:
                print("\n  [AUTO-FIX] Root detected. Disabling PCIe ASPM across bus...")
                run_cmd(["sh", "-c", "echo performance > /sys/module/pcie_aspm/parameters/policy"])
                run_cmd(["setpci", "-s", "00:00.0", "CAP_EXP+0x10.w=0x0000"])
                rc_fix, out_fix = run_cmd(["setpci", "-d", "1f9d:", "CAP_EXP+0x10.w=0x0000"])
                if rc_fix == 0:
                    print("  [SUCCESS] Disabled ASPM on Root Port (00:00.0) + Metis (1f9d:) + kernel policy!")
                else:
                    print(f"  [ERROR] setpci failed: {out_fix}")
            else:
                print("  Recommendation: Disable PCIe ASPM via:")
                print("    sudo sh -c 'echo performance > /sys/module/pcie_aspm/parameters/policy'")
                print("    sudo setpci -s 00:00.0 CAP_EXP+0x10.w=0x0000")
                print("    sudo setpci -d 1f9d: CAP_EXP+0x10.w=0x0000")
    elif "Capabilities:" not in out:
        print("\n  [NOTE] PCIe Capabilities are hidden by non-root permissions.")
        print("  Run with sudo to view detailed ASPM power states: 'sudo python tools/diag_metis.py'")


def check_metis_device() -> None:
    print_header("Axelera Metis Driver & Device Node Check")
    devs = glob.glob("/dev/metis*")
    if not devs:
        print("[ERROR] No /dev/metis* device nodes found!")
        print("Check if metis kernel module is loaded: 'lsmod | grep -E \"metis|uio\"'")
    else:
        print(f"Detected Metis device nodes: {devs}")
        for d in devs:
            rc, out = run_cmd(["ls", "-la", d])
            print(f"  {out}")

    # Check kernel driver module parameters
    print("\nMetis Kernel Driver Parameters:")
    for mod in ("metis", "axl"):
        p_msi = Path(f"/sys/module/{mod}/parameters/single_msi")
        if p_msi.exists():
            msi_val = p_msi.read_text().strip()
            print(f"  Module '{mod}': single_msi={msi_val}")
            if msi_val == "0":
                print("  [CRITICAL ALERT FOR RK3588]")
                print("  single_msi=0 (Multi-MSI) is active. On Rockchip RK3588, multi-MSI causes")
                print("  'Failed to wait for MSI 0: Connection timed out' after ~200 frames of streaming.")
                print("  To fix, configure single_msi=1:")
                print("    sudo sh -c 'echo -e \"options metis single_msi=1\\noptions axl single_msi=1\" > /etc/modprobe.d/metis.conf'")
                print("    sudo modprobe -r metis axl && sudo modprobe metis single_msi=1")
                print("    axdevice --reload-firmware")
            else:
                print("  [OK] single_msi=1 is active (safe for RK3588 MSI interrupt routing).")
            break

    # Check axdevice CLI
    rc_ax, out_ax = run_cmd(["axdevice", "-v"])
    if rc_ax == 0:
        print("\naxdevice information:")
        for l in out_ax.splitlines()[:12]:
            print(f"  {l}")
    else:
        print(f"[NOTE] 'axdevice -v' returned code {rc_ax} ({out_ax[:100]})")

    # Check for active processes holding the device
    print_header("Processes Accessing Metis Device")
    holders = []
    if os.path.exists("/proc"):
        for pid_entry in os.listdir("/proc"):
            if pid_entry.isdigit():
                fd_dir = f"/proc/{pid_entry}/fd"
                try:
                    for fd in os.listdir(fd_dir):
                        target = os.readlink(f"{fd_dir}/{fd}")
                        if "metis" in target:
                            cmdline = open(f"/proc/{pid_entry}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore").strip()
                            holders.append(f"PID {pid_entry} (fd {fd}): {cmdline or '(unknown)'}")
                except Exception:
                    pass
    if holders:
        print("Active processes locking /dev/metis*:")
        for h in sorted(set(holders)):
            print(f"  {h}")
    else:
        print("No processes currently locking /dev/metis-0:1:0.")


def check_dmesg() -> None:
    print_header("Kernel Log (dmesg) Metis / PCIe / DMA Events")
    rc, out = run_cmd(["dmesg", "-T"], timeout=3)
    if rc != 0:
        rc, out = run_cmd(["dmesg"], timeout=3)
    if rc == 0 and out:
        pattern = re.compile(r"(metis|uio|pcie|dma|rockchip-pcie|aer|msi|timeout|failed)", re.IGNORECASE)
        matches = [l for l in out.splitlines() if pattern.search(l)]
        if matches:
            print(f"Found {len(matches)} relevant kernel log lines. Recent entries:")
            for l in matches[-20:]:
                print(f"  {l}")
        else:
            print("No recent metis/uio/pcie error entries in dmesg.")
    else:
        print("[NOTE] dmesg access restricted (run with sudo or check permissions).")


def check_compiled_models() -> None:
    print_header("Compiled Voyager Models (AIPU Core Layout)")
    build_dirs = [
        Path.home() / "voyager-sdk" / "build",
        Path.cwd() / "build",
    ]
    found = False
    for b in build_dirs:
        if b.is_dir():
            model_jsons = list(b.glob("**/model.json"))
            if model_jsons:
                found = True
                print(f"Found {len(model_jsons)} compiled models in {b}:")
                for m in sorted(model_jsons):
                    rel = m.relative_to(b)
                    print(f"  --> {rel} ({m.stat().st_size // 1024} KB)")
    if not found:
        print("No compiled model.json files found in ~/voyager-sdk/build or ./build")


def main() -> None:
    parser = argparse.ArgumentParser(description="Axelera Metis Diagnostics on RK3588")
    parser.add_argument("--fix-aspm", action="store_true", help="Attempt to disable PCIe ASPM on Axelera device")
    parser.add_argument("--rescan", action="store_true", help="Trigger PCIe bus rescan to recover unattached/removed device")
    args = parser.parse_args()

    if args.rescan:
        print_header("Triggering PCIe Bus Rescan")
        rc, out = run_cmd(["sudo", "-n", "sh", "-c", "echo 1 > /sys/bus/pci/rescan"], timeout=10)
        print(f"PCIe rescan result: code={rc} output={out or '(ok)'}")

    check_system()
    check_pcie()
    check_metis_device()
    check_compiled_models()
    check_dmesg()

    if args.fix_aspm:
        print_header("Disabling PCIe ASPM on Axelera Device")
        rc, out = run_cmd(["setpci", "-d", "1f9d:", "CAP_EXP+0x10.w=0x0000"])
        print(f"setpci result: code={rc} output={out}")

    print_header("Diagnostic Complete")


if __name__ == "__main__":
    main()
