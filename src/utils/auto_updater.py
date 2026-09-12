"""
AutoUpdater: Automated GitHub Repository Update Checker, Scheduled Installer, and Application Restarter.
"""

import os
import sys
import time
import datetime
import threading
import subprocess
import yaml
from typing import Dict, Any, List, Optional, Callable

CHECK_INTERVALS_SEC = {
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800
}

class AutoUpdater:
    """Manages periodic GitHub repository checks, scheduled update installations, and application restarts."""

    def __init__(self, config: Dict[str, Any], config_path: str = "config/config.yaml", restart_callback: Optional[Callable[[], None]] = None):
        self.config_path = config_path
        self.restart_callback = restart_callback

        update_cfg = config.get("auto_update", {})
        self.enabled = update_cfg.get("enabled", True)
        self.branch = update_cfg.get("branch", "live")
        self.check_interval = update_cfg.get("check_interval", "daily").lower()
        if self.check_interval not in CHECK_INTERVALS_SEC:
            self.check_interval = "daily"
        self.install_time = update_cfg.get("install_time", "immediately")
        self.remote = update_cfg.get("remote", "origin")

        self.last_check_time: Optional[str] = None
        self.next_check_time_ts: float = time.time() + CHECK_INTERVALS_SEC[self.check_interval]
        self.next_check_time: str = self._format_timestamp(self.next_check_time_ts)

        self.current_commit: str = self._get_local_commit()
        self.latest_commit: Optional[str] = None
        self.update_available: bool = False
        self.pending_commits: List[str] = []
        self.status: str = "idle"
        self.error_message: Optional[str] = None

        self.lock = threading.Lock()
        self.is_running = True

        self.worker_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.worker_thread.start()

        print(f"[AUTO-UPDATER] Initialized (Branch: '{self.branch}', Interval: '{self.check_interval}', Install Time: '{self.install_time}', Enabled: {self.enabled})")

    @staticmethod
    def _format_timestamp(ts: float) -> str:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def _get_local_commit(self) -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def _get_active_branch(self) -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return "main"

    def check_for_updates(self) -> Dict[str, Any]:
        """Queries GitHub remote for latest commits on configured branch."""
        with self.lock:
            self.status = "checking"
            self.error_message = None
            self.last_check_time = self._format_timestamp(time.time())

        target_branch = self.branch
        print(f"[AUTO-UPDATER] Checking remote '{self.remote}' on branch '{target_branch}'...")

        try:
            # 1. Fetch remote branch
            fetch_cmd = ["git", "fetch", self.remote, target_branch]
            fetch_res = subprocess.run(fetch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            if fetch_res.returncode != 0:
                # If target branch (e.g. 'live') was not found on remote, attempt fallback to 'main'
                fallback_branch = "main"
                print(f"[AUTO-UPDATER WARNING] Branch '{target_branch}' not found on remote. Trying fallback branch '{fallback_branch}'...")
                fallback_res = subprocess.run(["git", "fetch", self.remote, fallback_branch], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
                if fallback_res.returncode == 0:
                    target_branch = fallback_branch
                else:
                    err_msg = fetch_res.stderr.strip() or "Could not fetch from remote."
                    with self.lock:
                        self.status = "error"
                        self.error_message = err_msg
                    return {"success": False, "error": err_msg}

            # 2. Get local and remote HEAD commits
            local_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
            remote_ref = f"{self.remote}/{target_branch}"
            remote_hash = subprocess.check_output(["git", "rev-parse", remote_ref], text=True, timeout=5).strip()

            short_local = local_hash[:7]
            short_remote = remote_hash[:7]

            # 3. Check for commits ahead on remote
            log_res = subprocess.run(
                ["git", "log", f"HEAD..{remote_ref}", "--oneline", "-n", "10"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
            )
            pending = [line.strip() for line in log_res.stdout.strip().split("\n") if line.strip()]

            with self.lock:
                self.current_commit = short_local
                self.latest_commit = short_remote
                self.pending_commits = pending
                self.update_available = len(pending) > 0 or (local_hash != remote_hash)
                self.status = "update_available" if self.update_available else "idle"
                self.next_check_time_ts = time.time() + CHECK_INTERVALS_SEC.get(self.check_interval, 86400)
                self.next_check_time = self._format_timestamp(self.next_check_time_ts)

            print(f"[AUTO-UPDATER] Check complete. Current: {short_local}, Remote ({target_branch}): {short_remote}. Update available: {self.update_available} ({len(pending)} commits)")
            return {
                "success": True,
                "update_available": self.update_available,
                "current_commit": short_local,
                "latest_commit": short_remote,
                "pending_commits": pending,
                "branch_checked": target_branch
            }

        except Exception as e:
            err_str = str(e)
            print(f"[AUTO-UPDATER ERROR] Check failed: {err_str}")
            with self.lock:
                self.status = "error"
                self.error_message = err_str
                self.next_check_time_ts = time.time() + CHECK_INTERVALS_SEC.get(self.check_interval, 86400)
                self.next_check_time = self._format_timestamp(self.next_check_time_ts)
            return {"success": False, "error": err_str}

    def apply_update_and_restart(self) -> Dict[str, Any]:
        """Pulls latest commits from GitHub and triggers application restart."""
        with self.lock:
            self.status = "installing"
            self.error_message = None

        target_branch = self.branch
        print(f"[AUTO-UPDATER] Pulling latest updates from {self.remote}/{target_branch}...")

        try:
            pull_cmd = ["git", "pull", self.remote, target_branch]
            pull_res = subprocess.run(pull_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            
            if pull_res.returncode != 0:
                # Attempt fallback branch if configured branch failed
                fallback_branch = "main"
                print(f"[AUTO-UPDATER WARNING] Pull on '{target_branch}' failed. Attempting fallback '{fallback_branch}'...")
                pull_res = subprocess.run(["git", "pull", self.remote, fallback_branch], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
                if pull_res.returncode == 0:
                    target_branch = fallback_branch
                else:
                    err_msg = pull_res.stderr.strip() or "git pull failed"
                    with self.lock:
                        self.status = "error"
                        self.error_message = err_msg
                    return {"success": False, "error": err_msg}

            new_commit = self._get_local_commit()
            print(f"[AUTO-UPDATER SUCCESS] Successfully updated codebase to commit {new_commit}. Triggering restart...")

            with self.lock:
                self.current_commit = new_commit
                self.latest_commit = new_commit
                self.update_available = False
                self.pending_commits = []
                self.status = "restarting"

            # Execute restart in a deferred thread so API caller gets a successful response
            def deferred_restart():
                time.sleep(1.5)
                self._execute_restart()

            threading.Thread(target=deferred_restart, daemon=True).start()

            return {
                "success": True,
                "message": f"Successfully pulled update to commit {new_commit}. System is restarting.",
                "commit": new_commit
            }

        except Exception as e:
            err_str = str(e)
            print(f"[AUTO-UPDATER ERROR] Update installation failed: {err_str}")
            with self.lock:
                self.status = "error"
                self.error_message = err_str
            return {"success": False, "error": err_str}

    def _execute_restart(self):
        """Executes application restart."""
        print("==========================================================")
        print("  [AUTO-UPDATER] Restarting Axelera Metis Application...   ")
        print("==========================================================")
        
        if self.restart_callback:
            try:
                self.restart_callback()
                return
            except Exception as e:
                print(f"[AUTO-UPDATER] Restart callback error: {e}")

        # Fallback direct re-exec
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    def _update_loop(self):
        """Periodic background thread to monitor intervals and preferred installation times."""
        # Initial wait before running first check (10 seconds after boot)
        time.sleep(10)

        while self.is_running:
            try:
                now = time.time()
                current_dt = datetime.datetime.now()

                # Condition 1: Perform periodic check
                if self.enabled and now >= self.next_check_time_ts:
                    self.check_for_updates()

                # Condition 2: Check if an update is available and should be installed
                if self.enabled and self.update_available and self.status not in ["installing", "restarting"]:
                    if self.install_time == "immediately":
                        print("[AUTO-UPDATER] Update detected with 'immediately' policy. Applying update...")
                        self.apply_update_and_restart()
                    else:
                        # Parse HH:MM
                        try:
                            parts = self.install_time.split(":")
                            target_hour = int(parts[0])
                            target_min = int(parts[1]) if len(parts) > 1 else 0
                            if current_dt.hour == target_hour and current_dt.minute == target_min:
                                print(f"[AUTO-UPDATER] Scheduled maintenance window reached ({self.install_time}). Applying update...")
                                self.apply_update_and_restart()
                        except Exception:
                            pass

            except Exception as e:
                print(f"[AUTO-UPDATER WORKER WARNING] Loop error: {e}")

            time.sleep(30)  # Check every 30 seconds

    def update_config(self, new_cfg: Dict[str, Any]) -> bool:
        """Updates runtime configuration and saves to YAML config file."""
        with self.lock:
            if "enabled" in new_cfg:
                self.enabled = bool(new_cfg["enabled"])
            if "check_interval" in new_cfg and new_cfg["check_interval"] in CHECK_INTERVALS_SEC:
                self.check_interval = new_cfg["check_interval"]
                self.next_check_time_ts = time.time() + CHECK_INTERVALS_SEC[self.check_interval]
                self.next_check_time = self._format_timestamp(self.next_check_time_ts)
            if "install_time" in new_cfg:
                self.install_time = str(new_cfg["install_time"]).strip()
            if "branch" in new_cfg and new_cfg["branch"]:
                self.branch = str(new_cfg["branch"]).strip()

        # Persist to YAML file
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    full_cfg = yaml.safe_load(f) or {}

                if "auto_update" not in full_cfg:
                    full_cfg["auto_update"] = {}

                full_cfg["auto_update"]["enabled"] = self.enabled
                full_cfg["auto_update"]["check_interval"] = self.check_interval
                full_cfg["auto_update"]["install_time"] = self.install_time
                full_cfg["auto_update"]["branch"] = self.branch
                full_cfg["auto_update"]["remote"] = self.remote

                with open(self.config_path, "w") as f:
                    yaml.safe_dump(full_cfg, f, sort_keys=False)

                print(f"[AUTO-UPDATER] Config successfully saved to {self.config_path}")
                return True
        except Exception as e:
            print(f"[AUTO-UPDATER ERROR] Failed to save config to {self.config_path}: {e}")
            return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Returns complete telemetry dictionary for Admin Panel UI."""
        with self.lock:
            return {
                "enabled": self.enabled,
                "branch": self.branch,
                "check_interval": self.check_interval,
                "install_time": self.install_time,
                "remote": self.remote,
                "current_commit": self.current_commit,
                "latest_commit": self.latest_commit,
                "update_available": self.update_available,
                "pending_commits": self.pending_commits,
                "last_check_time": self.last_check_time or "Not checked yet",
                "next_check_time": self.next_check_time,
                "status": self.status,
                "error_message": self.error_message
            }

    def stop(self):
        """Stops background worker thread."""
        self.is_running = False
