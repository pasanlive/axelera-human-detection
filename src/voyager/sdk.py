"""
Lazy import guard for the Axelera Voyager SDK.

The SDK is only available on the Metis host after ``source venv/bin/activate``
inside the voyager-sdk checkout. We never silently fall back to CPU inference:
production code calls :func:`require_voyager`, which raises with actionable
instructions when the SDK is missing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

_IMPORT_ERROR: Optional[BaseException] = None


@lru_cache(maxsize=1)
def _try_import() -> bool:
    global _IMPORT_ERROR
    try:
        import axelera.app.stream  # noqa: F401
        return True
    except Exception as e:  # ImportError, or SDK runtime init failures
        _IMPORT_ERROR = e
        return False


def voyager_available() -> bool:
    """True when ``axelera.app`` can be imported in this interpreter."""
    return _try_import()


def require_voyager() -> None:
    """Raises RuntimeError with setup instructions if the SDK is unavailable."""
    if voyager_available():
        return
    raise RuntimeError(
        "Axelera Voyager SDK (axelera.app) is not importable: "
        f"{_IMPORT_ERROR!r}.\n"
        "  1. On the Metis host: cd <voyager-sdk> && source venv/bin/activate\n"
        "  2. Ensure AXELERA_FRAMEWORK points at the SDK checkout.\n"
        "  3. Run `python tools/deploy_networks.py` once to quantize/compile the networks.\n"
        "  For UI development without hardware use: python app.py --simulate"
    )


def framework_path(configured: Optional[str] = None) -> Optional[str]:
    """Resolves the Voyager SDK checkout (contains deploy.py)."""
    for cand in (configured, os.environ.get("AXELERA_FRAMEWORK")):
        if cand and os.path.isfile(os.path.join(os.path.expanduser(cand), "deploy.py")):
            return os.path.abspath(os.path.expanduser(cand))
    return None


DEFAULT_NETWORKS_SUBDIR = "customers/upside"


def resolve_networks_dir(configured: Optional[str], sdk_path: Optional[str] = None,
                         fallback_root: Optional[str] = None) -> "Path":
    """Where the generated network YAMLs are written.

    Voyager 1.6.0 ``deploy.py`` only accepts networks registered by scanning YAMLs
    *inside* the SDK checkout (``ax_models/``, ``customers/`` ...); a YAML anywhere
    else fails with "Invalid network ... no close match found". So relative paths
    are resolved against ``$AXELERA_FRAMEWORK`` (default ``customers/upside``).
    Without an SDK (dev laptop / --dry-run) we fall back to ``<project>/voyager/networks``.
    """
    from pathlib import Path

    fw = framework_path(sdk_path)
    rel = configured or DEFAULT_NETWORKS_SUBDIR
    if rel == "voyager/networks":  # legacy default (outside the SDK tree => unusable by deploy.py)
        rel = DEFAULT_NETWORKS_SUBDIR
    p = Path(os.path.expanduser(rel))
    if p.is_absolute():
        return p
    if fw:
        return (Path(fw) / p).resolve()
    return (Path(fallback_root or os.getcwd()) / "voyager" / "networks").resolve()


def sdk_modules() -> Any:
    """Returns a namespace with the SDK modules the runtime needs (imports lazily)."""
    require_voyager()
    from types import SimpleNamespace

    from axelera.app import config as ax_config
    from axelera.app import inf_tracers
    from axelera.app.stream import create_inference_stream

    try:
        from axelera.app.pipe import FrameEventType
    except ImportError:  # pragma: no cover - older layout
        from axelera.app.pipe.frame_data import FrameEventType

    return SimpleNamespace(
        config=ax_config,
        inf_tracers=inf_tracers,
        create_inference_stream=create_inference_stream,
        FrameEventType=FrameEventType,
    )
