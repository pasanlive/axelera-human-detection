"""
Daily Rolling Logger with 30-Day Retention Management.

Features:
- Daily rolling log files: app_YYYY-MM-DD.log stored in data/logs/
- Automatic removal of logs older than 30 days (configurable retention)
- Captures stdout, stderr, and Python logging into the active daily log file
- Thread-safe writes with automatic midnight rotation
- Utility methods to list, view, and stream logs for the Web Dashboard API
"""

import datetime
import glob
import logging
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_LOG_DIR = "data/logs"
DEFAULT_RETENTION_DAYS = 30
LOG_FILENAME_PATTERN = re.compile(r"^app_(\d{4}-\d{2}-\d{2})\.log$")


def _format_size(size_bytes: int) -> str:
    """Formats bytes into human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


class DailyRollingLogger:
    """
    Manages daily rolling log files and automatic retention pruning.
    Thread-safe dual-stream output preserving terminal prints while capturing to disk.
    """

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR, max_retention_days: int = DEFAULT_RETENTION_DAYS,
                 capture_stdout: bool = True):
        self.log_dir = os.path.abspath(log_dir)
        self.max_retention_days = max(1, int(max_retention_days))
        self.capture_stdout = capture_stdout
        self.lock = threading.Lock()

        os.makedirs(self.log_dir, exist_ok=True)

        self.current_date: Optional[str] = None
        self.current_filepath: Optional[str] = None
        self.file_handle: Optional[Any] = None

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._is_installed = False

        # Open initial daily log and purge expired files
        self._rotate_if_needed()
        self.cleanup_old_logs()

        if self.capture_stdout:
            self.install_stream_capture()

    def _rotate_if_needed(self):
        """Checks if date has rolled over and opens/creates the corresponding daily log."""
        today = datetime.date.today().isoformat()
        if today != self.current_date or self.file_handle is None or self.file_handle.closed:
            if self.file_handle and not self.file_handle.closed:
                try:
                    self.file_handle.flush()
                    self.file_handle.close()
                except Exception:
                    pass

            self.current_date = today
            self.current_filepath = os.path.join(self.log_dir, f"app_{today}.log")
            try:
                self.file_handle = open(self.current_filepath, "a", encoding="utf-8", buffering=1)
            except Exception as e:
                self._original_stderr.write(f"[LOGGER ERROR] Failed to open daily log {self.current_filepath}: {e}\n")
                self.file_handle = None

            # Run retention cleanup on daily rollover
            try:
                self.cleanup_old_logs()
            except Exception:
                pass

    def write_record(self, text: str, stream_name: str = "stdout"):
        """Writes text to both the daily rolling log and the console."""
        with self.lock:
            self._rotate_if_needed()
            if self.file_handle:
                try:
                    # Append timestamp prefix if line doesn't already contain one
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    for line in text.splitlines(keepends=True):
                        # Avoid double timestamping if already formatted
                        if line.startswith("[20") and len(line) > 20 and line[10] in (" ", "T"):
                            formatted = line
                        else:
                            formatted = f"[{now_str}] {line}" if line.strip() else line
                        self.file_handle.write(formatted)
                    self.file_handle.flush()
                except Exception:
                    pass

        # Echo to original console
        target = self._original_stderr if stream_name == "stderr" else self._original_stdout
        try:
            target.write(text)
            target.flush()
        except Exception:
            pass

    def install_stream_capture(self):
        """Hooks sys.stdout and sys.stderr to capture all console output."""
        if self._is_installed:
            return

        class StreamTee:
            def __init__(tee_self, logger_ref: "DailyRollingLogger", stream_name: str, original_stream):
                tee_self.logger = logger_ref
                tee_self.stream_name = stream_name
                tee_self.original = original_stream

            def write(tee_self, s):
                tee_self.logger.write_record(s, stream_name=tee_self.stream_name)

            def flush(tee_self):
                try:
                    tee_self.original.flush()
                except Exception:
                    pass

            def fileno(tee_self):
                return getattr(tee_self.original, "fileno", lambda: 1)()

            def isatty(tee_self):
                return getattr(tee_self.original, "isatty", lambda: False)()

        sys.stdout = StreamTee(self, "stdout", self._original_stdout)
        sys.stderr = StreamTee(self, "stderr", self._original_stderr)
        self._is_installed = True

    def uninstall_stream_capture(self):
        """Restores original sys.stdout and sys.stderr."""
        if not self._is_installed:
            return
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._is_installed = False

    def cleanup_old_logs(self, max_days: Optional[int] = None) -> List[str]:
        """
        Removes any log file in log_dir older than max_days.
        Default is 30 days retention.
        """
        days = max_days if max_days is not None else self.max_retention_days
        now = time.time()
        cutoff_sec = days * 86400.0
        removed = []

        try:
            for filepath in glob.glob(os.path.join(self.log_dir, "*.log")):
                filename = os.path.basename(filepath)
                match = LOG_FILENAME_PATTERN.match(filename)
                should_remove = False

                if match:
                    # Check by parsed date
                    try:
                        log_date = datetime.date.fromisoformat(match.group(1))
                        age_days = (datetime.date.today() - log_date).days
                        if age_days > days:
                            should_remove = True
                    except Exception:
                        pass

                # Fallback to file modification time
                if not should_remove:
                    try:
                        mtime = os.path.getmtime(filepath)
                        if now - mtime > cutoff_sec:
                            should_remove = True
                    except Exception:
                        pass

                if should_remove:
                    try:
                        os.remove(filepath)
                        removed.append(filename)
                    except Exception as e:
                        self._original_stderr.write(f"[LOGGER NOTICE] Could not delete expired log {filename}: {e}\n")
        except Exception as e:
            self._original_stderr.write(f"[LOGGER NOTICE] Error during log retention pruning: {e}\n")

        return removed

    def list_logs(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available log files sorted by date descending."""
        logs = []
        today = datetime.date.today().isoformat()

        try:
            for filepath in glob.glob(os.path.join(self.log_dir, "*.log")):
                filename = os.path.basename(filepath)
                size_bytes = 0
                mtime = 0.0
                try:
                    stat = os.stat(filepath)
                    size_bytes = stat.st_size
                    mtime = stat.st_mtime
                except Exception:
                    pass

                match = LOG_FILENAME_PATTERN.match(filename)
                date_str = match.group(1) if match else time.strftime("%Y-%m-%d", time.localtime(mtime))
                is_current = (date_str == today) or (filename == f"app_{today}.log")

                logs.append({
                    "filename": filename,
                    "date": date_str,
                    "size_bytes": size_bytes,
                    "size_formatted": _format_size(size_bytes),
                    "modified": mtime,
                    "modified_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                    "is_current": is_current,
                })
        except Exception:
            pass

        # Sort newest first
        logs.sort(key=lambda x: (x["date"], x["modified"]), reverse=True)
        return logs

    def read_log(self, filename: Optional[str] = None, tail: Optional[int] = 1000) -> Dict[str, Any]:
        """Reads log file contents with optional line tailing."""
        today = datetime.date.today().isoformat()
        if not filename or filename in ("current", f"app_{today}.log"):
            filename = f"app_{today}.log"

        safe_filename = os.path.basename(filename)
        filepath = os.path.join(self.log_dir, safe_filename)

        if not os.path.isfile(filepath):
            return {
                "error": f"Log file '{safe_filename}' not found",
                "filename": safe_filename,
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "is_current": False,
            }

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            if tail and tail > 0 and total_lines > tail:
                returned_lines = lines[-tail:]
            else:
                returned_lines = lines

            content = "".join(returned_lines)
            size = os.path.getsize(filepath)

            return {
                "filename": safe_filename,
                "content": content,
                "total_lines": total_lines,
                "lines_returned": len(returned_lines),
                "size_bytes": size,
                "size_formatted": _format_size(size),
                "is_current": (safe_filename == f"app_{today}.log"),
            }
        except Exception as e:
            return {
                "error": str(e),
                "filename": safe_filename,
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "is_current": False,
            }

    def get_log_filepath(self, filename: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Returns safe verified absolute path and filename for downloading."""
        today = datetime.date.today().isoformat()
        if not filename or filename in ("current", f"app_{today}.log"):
            filename = f"app_{today}.log"

        safe_filename = os.path.basename(filename)
        filepath = os.path.join(self.log_dir, safe_filename)

        if os.path.isfile(filepath):
            return filepath, safe_filename
        return None, None


# Global singleton instance
_GLOBAL_LOGGER: Optional[DailyRollingLogger] = None


def setup_daily_logger(log_dir: str = DEFAULT_LOG_DIR, max_retention_days: int = DEFAULT_RETENTION_DAYS,
                       capture_stdout: bool = True) -> DailyRollingLogger:
    """Initializes global DailyRollingLogger singleton."""
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = DailyRollingLogger(log_dir=log_dir, max_retention_days=max_retention_days,
                                            capture_stdout=capture_stdout)
    return _GLOBAL_LOGGER


def get_logger() -> DailyRollingLogger:
    """Gets or initializes the global DailyRollingLogger."""
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = setup_daily_logger()
    return _GLOBAL_LOGGER
