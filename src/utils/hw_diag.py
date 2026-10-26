"""
Hardware and System Diagnostic Watchdog for Axelera Metis and Linux/Rockchip platforms.

Provides:
- faulthandler and SIGABRT/SIGSEGV crash handling (captures stack traces and kernel dmesg on C++ crashes)
- Real-time kernel dmesg monitoring (detects PCIe MSI interrupts, UIO driver, and DMA transfer errors)
- Inference heartbeat monitoring (detects pipeline stalls / AIPU kernel timeouts before abort)
- PCIe link and board telemetry inspection (ASP, link width, clock, temperatures)
"""

from __future__ import annotations

import faulthandler
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional


_WATCHDOG_INSTANCE: Optional[HardwareWatchdog] = None


def get_watchdog() -> HardwareWatchdog:
    """Returns the global HardwareWatchdog instance."""
    global _WATCHDOG_INSTANCE
    if _WATCHDOG_INSTANCE is None:
        _WATCHDOG_INSTANCE = HardwareWatchdog()
    return _WATCHDOG_INSTANCE


def enable_crash_handler(on_crash_callback: Optional[Callable[[], None]] = None) -> None:
    """Enables Python faulthandler and attaches signal diagnostic dumps for C++ aborts.

    When Axelera Level-Zero throws L0::Exception or abort() is called, this dumps:
    1. Python and C thread backtraces
    2. Linux kernel dmesg tail (Metis / UIO / PCIe errors)
    3. Last recorded inference heartbeat state
    """
    try:
        faulthandler.enable(all_threads=True)
    except Exception as e:
        print(f"[DIAG] Could not enable faulthandler: {e}", file=sys.stderr)

    def _sig_handler(sig_num: int, frame: Any) -> None:
        sig_name = signal.Signals(sig_num).name if hasattr(signal, "Signals") else str(sig_num)
        print(f"\n{'='*70}\n[FATAL CRASH DETECTED] Signal {sig_name} ({sig_num}) received!\n{'='*70}", file=sys.stderr)

        wd = get_watchdog()
        metrics = wd.get_metrics()
        print(f"[DIAG STATE] Frames processed: {metrics.get('frames_processed', 0)} | "
              f"Last frame latency: {metrics.get('last_latency_ms', 0):.1f}ms | "
              f"Stalled: {metrics.get('is_stalled', False)} ({metrics.get('stall_seconds', 0):.1f}s since last frame)",
              file=sys.stderr)

        print("\n[KERNEL DIAG] Recent kernel messages (dmesg):", file=sys.stderr)
        dmesg_lines = wd.get_dmesg_tail(25)
        if dmesg_lines:
            for line in dmesg_lines:
                print(f"  {line.strip()}", file=sys.stderr)
        else:
            print("  (dmesg unavailable or empty)", file=sys.stderr)

        print("\n[METIS PROCESSES] Active processes holding /dev/metis*:", file=sys.stderr)
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
            for h in sorted(set(holders)):
                print(f"  {h}", file=sys.stderr)
        else:
            print("  (None detected)", file=sys.stderr)

        if on_crash_callback:
            try:
                on_crash_callback()
            except Exception:
                pass

        print(f"{'='*70}\nTerminating process via default handler...\n{'='*70}\n", file=sys.stderr)
        # Restore default handler and re-raise to produce core dump / standard exit code
        signal.signal(sig_num, signal.SIG_DFL)
        os.kill(os.getpid(), sig_num)

    for sig in (signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGFPE):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            pass


class HardwareWatchdog:
    """Monitors hardware health, PCIe/UIO kernel errors, and inference heartbeat."""

    def __init__(self, stall_threshold_s: float = 3.0):
        self.stall_threshold_s = stall_threshold_s
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Heartbeat tracking
        self.total_frames = 0
        self.last_frame_time = 0.0
        self.last_cam_id: Optional[str] = None
        self.last_latency_ms = 0.0
        self.latencies: List[float] = []
        self.is_stalled = False
        self.stream_active = False

        # Dmesg tracking
        self._last_dmesg_seen: set = set()
        self._initial_dmesg_logged = False

    def disable_pcie_aspm(self) -> bool:
        """Disables PCIe ASPM on Axelera Metis (1f9d:1100), Root Port, and kernel policy to prevent RK3588 MSI timeout stalls."""
        # 1. Global Linux kernel ASPM policy -> performance (disables L0s/L1 system-wide)
        try:
            res_pol = subprocess.run(["sudo", "-n", "sh", "-c", "echo performance > /sys/module/pcie_aspm/parameters/policy"],
                                     capture_output=True, text=True, timeout=2)
            if res_pol.returncode == 0:
                print("[PCIe OPTIMIZATION] Set Linux PCIe ASPM policy to 'performance'.")
        except Exception:
            pass

        # 2. Disable ASPM on Metis endpoint (1f9d:)
        try:
            subprocess.run(["setpci", "-d", "1f9d:", "CAP_EXP+0x10.w=0x0000"],
                           capture_output=True, text=True, timeout=2)
            res_sudo = subprocess.run(["sudo", "-n", "setpci", "-d", "1f9d:", "CAP_EXP+0x10.w=0x0000"],
                                      capture_output=True, text=True, timeout=2)
            if res_sudo.returncode == 0:
                print("[PCIe OPTIMIZATION] Successfully disabled PCIe ASPM on Metis device via sudo setpci.")
        except Exception:
            pass

        # 3. Disable ASPM on upstream PCIe Root Port (00:00.0)
        try:
            res_rp = subprocess.run(["sudo", "-n", "setpci", "-s", "00:00.0", "CAP_EXP+0x10.w=0x0000"],
                                    capture_output=True, text=True, timeout=2)
            if res_rp.returncode == 0:
                print("[PCIe OPTIMIZATION] Successfully disabled PCIe ASPM on Root Port (00:00.0).")
                return True
        except Exception:
            pass

        return False

    def optimize_network_buffers(self) -> None:
        """Increases Linux network socket receive buffers to prevent RTSP UDP packet drops."""
        try:
            res = subprocess.run(["sudo", "-n", "sysctl", "-w",
                                  "net.core.rmem_max=26214400",
                                  "net.core.rmem_default=2097152"],
                                 capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                print("[NETWORK OPTIMIZATION] Increased Linux UDP socket buffer limit (rmem_max=25MB).")
        except Exception:
            pass

    def start(self) -> None:
        """Starts the background diagnostic watchdog thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self.disable_pcie_aspm()
            self.optimize_network_buffers()
            self._thread = threading.Thread(target=self._monitor_loop, name="hw-watchdog", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stops the watchdog."""
        with self._lock:
            self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_stream_active(self, active: bool) -> None:
        """Notifies the watchdog that the inference stream is actively processing."""
        with self._lock:
            self.stream_active = active
            if active and self.last_frame_time == 0.0:
                self.last_frame_time = time.time()

    def record_frame(self, cam_id: str, latency_ms: float = 0.0) -> None:
        """Records a successful frame output from the inference stream."""
        now = time.time()
        with self._lock:
            self.total_frames += 1
            self.last_frame_time = now
            self.last_cam_id = cam_id
            self.last_latency_ms = latency_ms
            self.is_stalled = False
            self.latencies.append(latency_ms)
            if len(self.latencies) > 100:
                self.latencies.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns snapshot of watchdog performance and stall telemetry."""
        now = time.time()
        with self._lock:
            stall_s = (now - self.last_frame_time) if self.last_frame_time > 0 else 0.0
            avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            p95_lat = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if self.latencies else 0.0
            return {
                "frames_processed": self.total_frames,
                "last_latency_ms": round(self.last_latency_ms, 1),
                "avg_latency_ms": round(avg_lat, 1),
                "p95_latency_ms": round(p95_lat, 1),
                "is_stalled": self.is_stalled,
                "stall_seconds": round(stall_s, 1),
                "last_cam_id": self.last_cam_id,
            }

    def get_dmesg_tail(self, n: int = 30) -> List[str]:
        """Reads the latest lines from the Linux kernel ring buffer."""
        try:
            cmd = ["dmesg", "-T"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if res.returncode != 0:
                res = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout:
                return res.stdout.strip().splitlines()[-n:]
        except Exception:
            pass
        return []

    def check_new_kernel_events(self) -> List[str]:
        """Queries dmesg for recent entries related to metis, uio, pcie, or dma."""
        lines = self.get_dmesg_tail(50)
        new_events = []
        pattern = re.compile(r"(metis|uio|pcie|dma|rockchip-pcie|aer|interrupt|msi|timeout|error|fail)", re.IGNORECASE)
        for line in lines:
            line_str = line.strip()
            if line_str and line_str not in self._last_dmesg_seen:
                self._last_dmesg_seen.add(line_str)
                if pattern.search(line_str):
                    new_events.append(line_str)
        # Prevent unbounded set growth
        if len(self._last_dmesg_seen) > 500:
            self._last_dmesg_seen = set(lines)
        return new_events

    def get_pcie_info(self) -> Dict[str, Any]:
        """Queries lspci for Axelera PCIe device information (Vendor ID 1f9d)."""
        info: Dict[str, Any] = {"found": False, "details": []}
        try:
            res = subprocess.run(["lspci", "-d", "1f9d:", "-vvv"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                info["found"] = True
                lines = res.stdout.strip().splitlines()
                info["details"] = lines
                for l in lines:
                    l_str = l.strip()
                    if "LnkCap:" in l_str or "LnkSta:" in l_str or "ASPM" in l_str:
                        info.setdefault("link_status", []).append(l_str)
                    if "DevSta:" in l_str or "CorrErr" in l_str or "UncorrErr" in l_str:
                        info.setdefault("errors", []).append(l_str)
        except Exception as e:
            info["error"] = str(e)
        return info

    def get_system_thermals(self) -> Dict[str, float]:
        """Reads Linux sysfs thermal zones."""
        temps: Dict[str, float] = {}
        tz_base = "/sys/class/thermal"
        if not os.path.isdir(tz_base):
            return temps
        try:
            for item in os.listdir(tz_base):
                if item.startswith("thermal_zone"):
                    z_dir = os.path.join(tz_base, item)
                    type_file = os.path.join(z_dir, "type")
                    temp_file = os.path.join(z_dir, "temp")
                    if os.path.isfile(type_file) and os.path.isfile(temp_file):
                        try:
                            z_type = open(type_file).read().strip()
                            z_temp = float(open(temp_file).read().strip()) / 1000.0
                            temps[z_type] = round(z_temp, 1)
                        except Exception:
                            pass
        except Exception:
            pass
        return temps

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        # Initial scan
        new_k = self.check_new_kernel_events()
        if new_k:
            print("[KERNEL INIT] Relevant kernel events on startup:")
            for l in new_k[-5:]:
                print(f"  {l}")

        while self._running:
            time.sleep(1.0)
            now = time.time()

            # 1. Check for kernel driver / PCIe errors
            new_events = self.check_new_kernel_events()
            for evt in new_events:
                print(f"[KERNEL DIAG] {evt}")

            # 2. Check for inference stream stalls
            with self._lock:
                if self.stream_active and self.last_frame_time > 0:
                    elapsed = now - self.last_frame_time
                    if elapsed >= self.stall_threshold_s:
                        if not self.is_stalled:
                            self.is_stalled = True
                            print(f"\n[WATCHDOG ALERT] Inference pipeline stall detected! "
                                  f"No frame received for {elapsed:.1f}s (threshold={self.stall_threshold_s}s).\n"
                                  f"  Last completed frame: #{self.total_frames} on '{self.last_cam_id}' "
                                  f"(latency: {self.last_latency_ms:.1f}ms). Checking hardware status...")

                            # Check dmesg immediately
                            urgent_k = self.check_new_kernel_events()
                            if urgent_k:
                                for l in urgent_k:
                                    print(f"  [KERNEL URGENT] {l}")
                            else:
                                print("  (No new kernel dmesg messages)")

                            # Check device node
                            if not os.path.exists("/dev/metis-0:1:0"):
                                print("  [CRITICAL] /dev/metis-0:1:0 device node has DISAPPEARED!")
                            else:
                                print("  /dev/metis-0:1:0 is present in /dev.")
