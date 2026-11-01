"""
Fault-Tolerant Process Supervisor and Hardware Auto-Recovery Engine for Upside AI.

Provides resilient process supervision for Axelera Metis embedded systems:
1. Wraps the main application worker in an isolated child process.
2. Intercepts native C++ uncatchable aborts (L0::Exception, -1301 wait kernel failure, SIGABRT, SIGSEGV).
3. Captures failure diagnostics and tail stderr into the dedicated daily rolling system log:
     ~/.upside-ai/logs/system_YYYY-MM-DD.log
4. Executes automated hardware recovery:
   - Kills rogue/orphaned axtrace processes and locks holding /dev/metis*
   - Verifies Linux PCIe link, ASPM policy, single_msi, and Metis device driver binding
   - Triggers PCIe rescan or module rebinding if the AIPU endpoint was detached
5. Performs autonomous restart with exponential backoff protection to maintain 24/7 continuous uptime.
6. Transparently proxies signals (SIGINT, SIGTERM) for graceful pipeline shutdown.
"""

from __future__ import annotations

import collections
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from src.utils.hw_diag import get_watchdog
from src.utils.system_logger import (
    record_status_change,
    record_system_crash,
    record_system_recovery,
    record_system_restart,
    record_system_start,
    record_system_stop,
)


class ProcessSupervisor:
    """Monitors worker process, logs system lifecycle changes, and auto-recovers on crashes."""

    def __init__(
        self,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        max_restarts: Optional[int] = None,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
    ):
        self.cmd = list(cmd)
        self.env = dict(env if env is not None else os.environ)
        self.max_restarts = max_restarts
        self.initial_backoff_s = max(0.1, float(initial_backoff_s))
        self.max_backoff_s = max(self.initial_backoff_s, float(max_backoff_s))

        self.restart_count = 0
        self.child_proc: Optional[subprocess.Popen] = None
        self.shutdown_requested = threading.Event()
        self.current_backoff = self.initial_backoff_s

        # Rolling ring buffer of output lines to parse crash reason
        self.output_buffer: collections.deque[str] = collections.deque(maxlen=150)
        self._buffer_lock = threading.Lock()

    def _stream_reader(self, pipe: Any, target_stream: Any) -> None:
        """Reads pipe line-by-line, forwards to target stream, and stores in rolling ring buffer."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                target_stream.write(line)
                target_stream.flush()
                clean = line.rstrip("\r\n")
                if clean:
                    with self._buffer_lock:
                        self.output_buffer.append(clean)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _extract_crash_reason(self) -> str:
        """Parses the recent console/stderr output buffer to determine root cause of crash."""
        with self._buffer_lock:
            lines = list(self.output_buffer)

        full_text = "\n".join(lines)
        # Priority root causes across entire diagnostic window
        if "Wait kernel failed with return code -1402" in full_text or "-1402" in full_text:
            return "Level-Zero wait kernel failure (-1402): Metis command queue / binary completion timeout"
        if "Wait kernel failed with return code -1301" in full_text or "-1301" in full_text:
            return "Level-Zero wait kernel timeout (-1301): AIPU command queue exceeded wait window"
        if "terminate called after throwing an instance of 'L0::Exception'" in full_text:
            return "Level-Zero C++ exception (L0::Exception) thrown from Metis driver thread"
        if "inplace_addstreamid: meta_key already exists" in full_text:
            return "GStreamer stream ID collision in libinplace_addstreamid.so"
        if "std::system_error" in full_text:
            return "C++ std::system_error: Invalid argument / device queue failure"
        if "Segmentation fault" in full_text or "SIGSEGV" in full_text:
            return "Process memory segmentation fault (SIGSEGV)"
        if "NetworkNotCompiledError" in full_text:
            return "Compiled network file (.ax64) not found for configured model"

        # Fallback to the last non-empty line
        for line in reversed(lines):
            clean = line.strip()
            if clean and not clean.startswith("[STATUS]") and not clean.startswith("[HINT]"):
                return clean[:140]

        return "Process aborted unexpectedly without stderr diagnostic"

    def recover_hardware(self) -> bool:
        """Executes hardware recovery steps to restore Metis AIPU and clear locks."""
        actions_taken = []
        success = True

        # 1. Terminate orphaned axtrace processes and rogue children
        try:
            res = subprocess.run(["pkill", "-9", "axtrace"], capture_output=True, timeout=3)
            if res.returncode == 0:
                actions_taken.append("killed orphaned axtrace")
        except Exception:
            pass

        # 2. Check and clear any processes holding /dev/metis*
        try:
            holders = []
            if os.path.exists("/proc"):
                for pid_entry in os.listdir("/proc"):
                    if pid_entry.isdigit() and int(pid_entry) != os.getpid():
                        fd_dir = f"/proc/{pid_entry}/fd"
                        try:
                            for fd in os.listdir(fd_dir):
                                target = os.readlink(f"{fd_dir}/{fd}")
                                if "metis" in target or "axl" in target:
                                    holders.append(int(pid_entry))
                                    break
                        except Exception:
                            pass
            for pid in set(holders):
                try:
                    os.kill(pid, signal.SIGKILL)
                    actions_taken.append(f"killed rogue pid {pid} holding /dev/metis")
                except Exception:
                    pass
        except Exception:
            pass

        # 3. Use HardwareWatchdog to verify device nodes, PCIe ASPM, single_msi, and driver binding
        try:
            wd = get_watchdog()
            wd_ok = wd.ensure_device_ready(force=True)
            if wd_ok:
                actions_taken.append("verified PCIe ASPM policy, single_msi=1, and device nodes")
            else:
                actions_taken.append("device readiness check flagged issues")
                success = False
        except Exception as e:
            actions_taken.append(f"watchdog verification failed: {e}")
            success = False

        summary = "; ".join(actions_taken) if actions_taken else "no recovery actions required"
        record_system_recovery(action="Hardware & PCIe Lock Cleanup", success=success, details=summary)
        return success

    def _signal_handler(self, sig_num: int, frame: Any) -> None:
        """Handles external termination signals (Ctrl+C, systemd stop)."""
        sig_name = signal.Signals(sig_num).name if hasattr(signal, "Signals") else str(sig_num)
        sys.stderr.write(f"\n[SUPERVISOR] Termination signal {sig_name} received. Stopping worker...\n")
        sys.stderr.flush()
        self.shutdown_requested.set()

        if self.child_proc and self.child_proc.poll() is None:
            try:
                self.child_proc.send_signal(signal.SIGINT)
            except Exception:
                pass

    def run(self) -> int:
        """Main supervisor loop: spawns, monitors, and recovers the child process."""
        # Install signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._signal_handler)

        sys.stderr.write("==========================================================\n")
        sys.stderr.write("  Upside AI Hardware Process Supervisor Active\n")
        sys.stderr.write("  Monitoring Axelera Metis Worker for Continuous Uptime\n")
        sys.stderr.write("==========================================================\n\n")
        sys.stderr.flush()

        while not self.shutdown_requested.is_set():
            child_env = dict(self.env)
            child_env["_UPSIDE_SUPERVISED"] = "1"

            # Clear output buffer for new run
            with self._buffer_lock:
                self.output_buffer.clear()

            t_start = time.time()
            try:
                self.child_proc = subprocess.Popen(
                    self.cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=child_env,
                )
            except Exception as e:
                err_msg = f"Failed to spawn worker process: {e}"
                sys.stderr.write(f"[SUPERVISOR ERROR] {err_msg}\n")
                record_system_crash(
                    pid=os.getpid(),
                    exit_code=-1,
                    signal_name="SPAWN_FAILED",
                    reason=err_msg,
                    uptime_s=0.0,
                )
                return 1

            # Start real-time stream forwarding threads
            t_out = threading.Thread(
                target=self._stream_reader,
                args=(self.child_proc.stdout, sys.stdout),
                name="sup-stdout",
                daemon=True,
            )
            t_err = threading.Thread(
                target=self._stream_reader,
                args=(self.child_proc.stderr, sys.stderr),
                name="sup-stderr",
                daemon=True,
            )
            t_out.start()
            t_err.start()

            # Wait for child to exit
            exit_code = self.child_proc.wait()
            t_out.join(timeout=2.0)
            t_err.join(timeout=2.0)
            uptime_s = time.time() - t_start

            # Case A: Normal clean shutdown requested or exit code 0
            if self.shutdown_requested.is_set():
                record_system_stop(reason="Supervisor clean shutdown requested by user / system signal")
                return 0

            if exit_code == 0:
                record_system_stop(reason=f"Worker process exited cleanly (exit code 0, uptime {uptime_s:.1f}s)")
                return 0

            # Case B: Crash / Abort detected!
            sig_name = "NONE"
            if exit_code < 0:
                sig_num = -exit_code
                sig_name = signal.Signals(sig_num).name if hasattr(signal, "Signals") and sig_num in signal.Signals._value2member_map_ else f"SIG_{sig_num}"
            elif exit_code > 128:
                sig_num = exit_code - 128
                sig_name = signal.Signals(sig_num).name if hasattr(signal, "Signals") and sig_num in signal.Signals._value2member_map_ else f"SIG_{sig_num}"

            reason = self._extract_crash_reason()
            sys.stderr.write(
                f"\n[SUPERVISOR ALERT] Worker PID {self.child_proc.pid} crashed with exit code {exit_code} "
                f"({sig_name}) after {uptime_s:.1f}s!\n"
                f"[SUPERVISOR ALERT] Diagnosed Reason: {reason}\n"
            )
            sys.stderr.flush()

            record_system_crash(
                pid=self.child_proc.pid,
                exit_code=exit_code,
                signal_name=sig_name,
                reason=reason,
                uptime_s=uptime_s,
            )

            # Check if max restarts exceeded
            self.restart_count += 1
            if self.max_restarts is not None and self.restart_count >= self.max_restarts:
                sys.stderr.write(f"[SUPERVISOR] Maximum restarts ({self.max_restarts}) reached. Terminating.\n")
                record_system_stop(reason=f"Maximum restart count ({self.max_restarts}) reached after crash")
                return exit_code

            # Execute hardware recovery
            sys.stderr.write("[SUPERVISOR RECOVERY] Executing hardware recovery & PCIe lock clearing...\n")
            self.recover_hardware()

            # Calculate backoff delay: if system ran stably for >60s, reset delay to initial_backoff_s
            if uptime_s > 60.0:
                self.current_backoff = self.initial_backoff_s
            else:
                self.current_backoff = min(self.max_backoff_s, self.current_backoff * 2.0)

            delay_s = self.current_backoff
            sys.stderr.write(f"[SUPERVISOR] Auto-restarting worker (attempt {self.restart_count}) in {delay_s:.1f}s...\n\n")
            record_system_restart(attempt=self.restart_count, delay_s=delay_s, reason=reason)

            # Interruptible sleep
            sleep_step = 0.1
            elapsed = 0.0
            while elapsed < delay_s and not self.shutdown_requested.is_set():
                time.sleep(sleep_step)
                elapsed += sleep_step

        record_system_stop(reason="Supervisor shutdown completed")
        return 0


def run_supervisor(cmd_args: Optional[List[str]] = None, max_restarts: Optional[int] = None) -> int:
    """Entry point for running under the supervisor."""
    python_bin = sys.executable
    app_script = os.path.abspath(sys.argv[0])
    full_cmd = [python_bin, app_script] + (cmd_args if cmd_args is not None else sys.argv[1:])
    supervisor = ProcessSupervisor(full_cmd, max_restarts=max_restarts)
    return supervisor.run()
