# Export main classes and types
from typing import Callable

from .laser_profiles import (
    GaussianBeam,
    LaserParameters,
    LaserProfile,
    WidefieldBeam,
)


# Define common power modulation patterns
def constant_power(power: float) -> Callable[[float], float]:
    """Create a constant power function."""
    return lambda t: power


# Version information
__version__ = "0.1.0"

# What to expose when using "from lasers import *"
__all__ = [
    "GaussianBeam",
    "LaserParameters",
    "LaserProfile",
    "constant_power",
    "WidefieldBeam",
]
