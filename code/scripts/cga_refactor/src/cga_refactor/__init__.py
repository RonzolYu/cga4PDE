"""CGA refactor public API."""

from .config import RunConfig, frozen_case_config
from .problems import make_problem
from .solver import run_cga


def run_campaign(*args, **kwargs):
    """Import the campaign entry lazily so ``python -m cga_refactor.run`` stays clean."""
    from .run import run_campaign as _run_campaign
    return _run_campaign(*args, **kwargs)

__all__ = ["RunConfig", "frozen_case_config", "make_problem", "run_cga", "run_campaign"]
__version__ = "0.1.0"
