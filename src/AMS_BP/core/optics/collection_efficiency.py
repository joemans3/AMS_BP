import warnings
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def collection_efficiency_isotropic_emission(na: float, n: float) -> float:
    ratio = na / n

    if ratio > 1.0:
        warnings.warn(
            f"NA ({na}) exceeds refractive index n ({n}), which is unphysical. "
            f"Clipping NA/n to 1.0 for collection efficiency calculation."
        )
        ratio = 1.0
    elif ratio < 0.0:
        warnings.warn(
            f"NA ({na}) is negative, which is unphysical. " f"Clipping NA/n to 0.0."
        )
        ratio = 0.0

    # Use identity: cos(arcsin(x)) = sqrt(1 - x^2)
    return 0.5 * (1 - np.sqrt(1 - ratio**2))
