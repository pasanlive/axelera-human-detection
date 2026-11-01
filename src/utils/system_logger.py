"""
Dedicated Daily Rolling System Lifecycle Event Logger.

Logs high-level system status transitions and health events into:
  ~/.upside-ai/logs/system_YYYY-MM-DD.log (or configured logs directory)

Events captured:
- [START]: Process startup, PID, active config, camera count, hardware backend
- [STATUS_CHANGE]: State transitions (e.g. camera connecting -> live, capacity tier changes)
- [CRASH]: Unhandled process crash / abort (exit code, signal name, failure reason, uptime)
- [RECOVERY]: Hardware recovery actions executed (clearing PCIe locks, killing zombies, resetting link)
- [RESTART]: Auto-restart triggered (attempt number, delay, reason)
- [STOP]: Clean shutdown requested (SIGTERM / SIGINT)
"""

from __future__ import annotations

import datetime
import glob
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SYSTEM_LOG_PATTERN = re.compile(r"^system_(\d{4}-\d{2}-\d{2})\.log$")


class SystemLogger:
    """Thread-safe dedicated system lifecycle logger with daily rotation."""

    def __init__(self, log_dir: Optional[str] = None, max_retention_days: int = 30):
        if log_dir is None:
            from src.utils.config_manager import UPSIDE_LOGS_DIR
            self.log_dir = os.path.abspath(UPSIDE_LOGS_DIR)
        else:
            self.log_dir = os.path.abspath(log_dir)

        self.max_retention_days = max(1, int(max_retention_days))
        self.lock = threading.Lock()
        os.makedirs(self.log_dir, exist_ok=True)

        self._current_date: Optional[str] = None
        self._current_filepath: Optional[str] = None
        self._file_handle: Optional[Any] = None

        self._rotate_if_needed()
        self.cleanup_old_logs()

    def _rotate_if_needed(self) -> None:
        today = datetime.date.today().isoformat()
        if today != self._current_date or self._file_handle is None or self._file_handle.closed:
            if self._file_handle and not self._file_handle.closed:
                try:
                    self._file_handle.flush()
                    self._file_handle.close()
                except Exception:
                    pass

            self._current_date = today
            self._current_filepath = os.path.join(self.log_dir, f"system_{today}.log")
            try:
                # Line buffered (buffering=1) for immediate persistence on disk
                self._file_handle = open(self._current_filepath, "a", encoding="utf-8", buffering=1)
            except Exception as e:
                sys.stderr.write(f"[SYSTEM LOGGER ERROR] Cannot open {self._current_filepath}: {e}\n")
                self._file_handle = None

            try:
                self.cleanup_old_logs()
            except Exception:
                pass

    def log(self, event_type: str, message: str, **kwargs: Any) -> str:
        """Appends a structured system event record to today's system log."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        event_tag = f"[{event_type.upper().strip()}]"

        extra_parts = []
        for k, v in kwargs.items():
            if v is not None:
                extra_parts.append(f"{k}={v}")
        extra_str = f" | {' '.join(extra_parts)}" if extra_parts else ""

        line = f"[{now_str}] {event_tag} {message}{extra_str}\n"

        with self.lock:
            self._rotate_if_needed()
            if self._file_handle:
                try:
                    self._file_handle.write(line)
                    self._file_handle.flush()
                    os.fsync(self._file_handle.fileno())
                except Exception as e:
                    sys.stderr.write(f"[SYSTEM LOGGER WRITE ERROR] {e}\n")

        # Also echo to stderr/stdout so supervisors or journalctl see it
        sys.stderr.write(f"[SYSTEM] {event_tag} {message}{extra_str}\n")
        sys.stderr.flush()
        return line

    def cleanup_old_logs(self) -> List[str]:
        """Prunes system_YYYY-MM-DD.log files older than max_retention_days."""
        deleted = []
        cutoff = datetime.date.today() - datetime.timedelta(days=self.max_retention_days)
        try:
            for p in glob.glob(os.path.join(self.log_dir, "system_*.log")):
                fname = os.path.basename(p)
                m = SYSTEM_LOG_PATTERN.match(fname)
                if m:
                    try:
                        file_date = datetime.date.fromisoformat(m.group(1))
                        if file_date < cutoff:
                            os.remove(p)
                            deleted.append(fname)
                    except Exception:
                        pass
        except Exception:
            pass
        return deleted

    def list_logs(self) -> List[Dict[str, Any]]:
        """Lists available system log files sorted by date descending."""
        res = []
        for p in glob.glob(os.path.join(self.log_dir, "system_*.log")):
            fname = os.path.basename(p)
            m = SYSTEM_LOG_PATTERN.match(fname)
            if m:
                try:
                    st = os.stat(p)
                    res.append({
                        "filename": fname,
                        "date": m.group(1),
                        "size_bytes": st.st_size,
                        "mtime": st.st_mtime,
                        "type": "system",
                    })
                except Exception:
                    pass
        res.sort(key=lambda x: x["date"], reverse=True)
        return res

    def read_log(self, date_str: Optional[str] = None, filename: Optional[str] = None, tail: int = 500) -> Dict[str, Any]:
        """Reads the tail lines of a specific system log file."""
        if filename:
            path = os.path.join(self.log_dir, os.path.basename(filename))
        elif date_str:
            path = os.path.join(self.log_dir, f"system_{date_str}.log")
        else:
            today = datetime.date.today().isoformat()
            path = os.path.join(self.log_dir, f"system_{today}.log")

        if not os.path.exists(path):
            return {"filename": os.path.basename(path), "exists": False, "lines": [], "total_lines": 0}

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            tail_lines = all_lines[-max(1, tail):] if len(all_lines) > tail else all_lines
            return {
                "filename": os.path.basename(path),
                "exists": True,
                "lines": [l.rstrip("\r\n") for l in tail_lines],
                "content": "".join(tail_lines),
                "total_lines": len(all_lines),
                "lines_returned": len(tail_lines),
            }
        except Exception as e:
            return {"filename": os.path.basename(path), "exists": False, "error": str(e), "lines": [], "content": ""}

    def get_log_filepath(self, filename: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Returns safe verified absolute path and filename for downloading."""
        today = datetime.date.today().isoformat()
        if not filename or filename in ("current", f"system_{today}.log"):
            filename = f"system_{today}.log"

        safe_filename = os.path.basename(filename)
        filepath = os.path.join(self.log_dir, safe_filename)
        if os.path.isfile(filepath):
            return filepath, safe_filename
        return None, None


# ── Global singleton accessor ────────────────────────────────────────────────

_SYSTEM_LOGGER_INSTANCE: Optional[SystemLogger] = None


def get_system_logger() -> SystemLogger:
    global _SYSTEM_LOGGER_INSTANCE
    if _SYSTEM_LOGGER_INSTANCE is None:
        _SYSTEM_LOGGER_INSTANCE = SystemLogger()
    return _SYSTEM_LOGGER_INSTANCE


# ── Convenience helper functions ─────────────────────────────────────────────

def log_system_event(event_type: str, message: str, **kwargs: Any) -> str:
    return get_system_logger().log(event_type, message, **kwargs)


def record_system_start(pid: int, active_config: str, camera_ids: List[str], backend: str = "metis_aipu", **kwargs: Any) -> str:
    cams_str = ",".join(camera_ids) if camera_ids else "none"
    return log_system_event("START", f"System started (PID {pid})",
                            config=active_config, cameras=cams_str, backend=backend, **kwargs)


def record_system_stop(reason: str = "Clean shutdown requested", **kwargs: Any) -> str:
    return log_system_event("STOP", f"System stopped: {reason}", **kwargs)


def record_system_crash(pid: int, exit_code: int, signal_name: str, reason: str, uptime_s: float, **kwargs: Any) -> str:
    uptime_fmt = f"{uptime_s:.1f}s"
    if uptime_s >= 60:
        mins = int(uptime_s // 60)
        secs = int(uptime_s % 60)
        uptime_fmt = f"{mins}m {secs}s"
    return log_system_event("CRASH", f"Process crashed (PID {pid})",
                            exit_code=exit_code, signal=signal_name, reason=reason, uptime=uptime_fmt, **kwargs)


def record_system_recovery(action: str, success: bool, details: str = "", **kwargs: Any) -> str:
    status = "SUCCESS" if success else "FAILED"
    return log_system_event("RECOVERY", f"Hardware recovery action [{status}]: {action}",
                            details=details or None, **kwargs)


def record_system_restart(attempt: int, delay_s: float, reason: str, **kwargs: Any) -> str:
    return log_system_event("RESTART", f"Auto-restarting system (attempt {attempt}) in {delay_s:.1f}s: {reason}",
                            attempt=attempt, delay=f"{delay_s:.1f}s", **kwargs)


def record_status_change(component: str, old_state: str, new_state: str, reason: str = "") -> str:
    msg = f"{component} state: {old_state} -> {new_state}"
    if reason:
        msg += f" ({reason})"
    return log_system_event("STATUS_CHANGE", msg, component=component, state=new_state)
