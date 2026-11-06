"""physXAI: physics-guided machine learning for MPC of building energy systems."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("physXAI")
except PackageNotFoundError:  # pragma: no cover - package is not installed (e.g. plain source checkout)
    __version__ = "0+unknown"

__all__ = ["__version__"]
