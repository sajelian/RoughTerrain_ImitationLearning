"""
Python module serving as a project/extension template.
"""

# Register Gym environments. Skipped if Isaac Sim isn't on PYTHONPATH so that the
# offline algorithms (e.g. FQL training) remain importable without launching sim.
try:
    from .tasks import *  # noqa: F401, F403
except ImportError as _e:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(
        f"RT_IL.tasks unavailable (likely no Isaac Sim on path): {_e}. "
        "Offline modules under RT_IL.algorithms remain available.",
        ImportWarning,
    )
