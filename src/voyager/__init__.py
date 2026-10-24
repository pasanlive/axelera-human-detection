"""
Voyager SDK integration layer.

Everything that touches the Axelera Voyager SDK (``axelera.app``) lives in this
package so the rest of the application stays importable on machines without the
SDK (developer laptops, CI) via :class:`~src.voyager.stream_runtime.SimulatedRuntime`.
"""

from src.voyager.sdk import voyager_available, require_voyager  # noqa: F401
