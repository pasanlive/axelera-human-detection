"""
Config and Environment Paths Manager for Upside AI Box.

Handles system-specific configuration files in ~/.upside-ai/config and
daily rolling logs in ~/.upside-ai/logs.
"""

import os
import shutil
from typing import Optional

UPSIDE_HOME = os.path.expanduser("~/.upside-ai")
UPSIDE_CONFIG_DIR = os.path.join(UPSIDE_HOME, "config")
UPSIDE_LOGS_DIR = os.path.join(UPSIDE_HOME, "logs")
DEFAULT_SYSTEM_CONFIG_PATH = os.path.join(UPSIDE_CONFIG_DIR, "config.yaml")


def get_upside_home() -> str:
    """Returns ~/.upside-ai base directory, creating it if needed."""
    os.makedirs(UPSIDE_HOME, exist_ok=True)
    return UPSIDE_HOME


def get_system_config_dir() -> str:
    """Returns ~/.upside-ai/config directory, creating it if needed."""
    os.makedirs(UPSIDE_CONFIG_DIR, exist_ok=True)
    return UPSIDE_CONFIG_DIR


def get_system_logs_dir() -> str:
    """Returns ~/.upside-ai/logs directory, creating it if needed."""
    os.makedirs(UPSIDE_LOGS_DIR, exist_ok=True)
    return UPSIDE_LOGS_DIR


def ensure_system_configs(project_config_dir: str = "config", custom_home: Optional[str] = None) -> str:
    """
    Checks for config files availability in ~/.upside-ai/config.
    If not available, creates config files based on project config files.
    Copies all template configs from project_config_dir.
    Returns path to active system config.yaml.
    """
    if custom_home:
        sys_config_dir = os.path.join(custom_home, "config")
    else:
        sys_config_dir = get_system_config_dir()

    os.makedirs(sys_config_dir, exist_ok=True)
    sys_config_path = os.path.join(sys_config_dir, "config.yaml")

    # If system config.yaml does not exist, copy from project config
    project_config_path = os.path.join(project_config_dir, "config.yaml")
    if not os.path.exists(sys_config_path):
        if os.path.exists(project_config_path):
            print(f"[UPSIDE CONFIG] First launch / missing config detected. Initializing {sys_config_path} from {project_config_path}...")
            shutil.copy2(project_config_path, sys_config_path)
        else:
            raise FileNotFoundError(f"Project template config not found at: {project_config_path}")

    # Synchronize any other configuration files from project_config_dir if missing in system dir
    if os.path.exists(project_config_dir):
        for item in os.listdir(project_config_dir):
            src_file = os.path.join(project_config_dir, item)
            dst_file = os.path.join(sys_config_dir, item)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"[UPSIDE CONFIG] Synchronized template config: {dst_file}")
                except Exception as e:
                    print(f"[UPSIDE CONFIG WARNING] Could not copy {src_file} to {dst_file}: {e}")

    return sys_config_path


def resolve_active_config_path(cli_config_arg: Optional[str] = None, project_config_dir: str = "config",
                               custom_home: Optional[str] = None) -> str:
    """
    Resolves active configuration path.
    If cli_config_arg was explicitly passed and is not the default "config/config.yaml", uses that path.
    Otherwise, ensures ~/.upside-ai/config files exist and returns ~/.upside-ai/config/config.yaml.
    """
    if cli_config_arg and cli_config_arg != "config/config.yaml":
        abs_p = os.path.abspath(cli_config_arg)
        if os.path.exists(abs_p):
            return abs_p

    return ensure_system_configs(project_config_dir=project_config_dir, custom_home=custom_home)
