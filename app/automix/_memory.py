"""Best-effort reclamation of unused allocator regions in this process only."""
from __future__ import annotations

from functools import lru_cache
from sys import platform as _platform


@lru_cache(maxsize=1)
def _pressure_relief():
    """Bind lazily; unsupported or unavailable APIs remain a cached no-op."""
    if _platform != "darwin":
        return None
    try:
        import ctypes

        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        function = library.malloc_zone_pressure_relief
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_size_t
        return function
    except Exception:  # binding is optional and must never prevent processing
        return None


def reclaim_unused() -> None:
    """Ask macOS to release unused pages; the native result is not an RSS count."""
    if _platform != "darwin":
        return
    try:
        function = _pressure_relief()
        if function is not None:
            function(None, 0)
    except Exception:  # a failed best-effort request must not alter DSP errors
        pass
