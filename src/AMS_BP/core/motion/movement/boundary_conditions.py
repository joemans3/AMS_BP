"""
Deprecated module due to switching to the Pyvista model for 3D cell shapes.
Removal Time: NDY (not determined yet)
"""

import numpy as np

from ....utils.decorators import _catch_recursion_error, deprecated

# Reflecting boundary condition which is a recursive function so that even if the first candidate
# is out of the space limit, the function will keep calling itself until the candidate is within the space limit


@deprecated(
    reason="Not used explicitly due to the use of Pyvista mesh objects to define shapes."
)
@_catch_recursion_error
def _refecting_boundary(
    fbm_store_last: float, fbm_candidate: float, space_lim: np.ndarray
) -> float:
    """Reflecting boundary condition for the FBM 1D

    Parameters:
    -----------
    fbm_store_last : float
        Last value of the FBM
    fbm_candidate : float
        Candidate value of the FBM
    space_lim : np.ndarray
        Space limit (min, max) for the FBM

    Returns:
    --------
    float
        New value of the FBM
    """
    if fbm_candidate > space_lim[1]:
        # if the candidate is greater than the space limit then reflect the difference back into the space limit
        return _refecting_boundary(
            fbm_store_last,
            space_lim[1] - np.abs(fbm_candidate - space_lim[1]),
            space_lim,
        )
    elif fbm_candidate < space_lim[0]:
        # if the candidate is less than the negative space limit then reflect the difference back into the space limit
        return _refecting_boundary(
            fbm_store_last,
            space_lim[0] + np.abs(fbm_candidate - space_lim[0]),
            space_lim,
        )
    else:
        return fbm_candidate


# Boundary condition where the step is set at the boundary limit if the candidate is out of the space limit


@deprecated(
    reason="Not used explicitly due to the use of Pyvista mesh objects to define shapes."
)
@_catch_recursion_error
def _absorbing_boundary(
    fbm_store_last: float, fbm_candidate: float, space_lim: np.ndarray
) -> float:
    """Absorbing boundary condition for the FBM 1D

    Parameters:
    -----------
    fbm_store_last : float
        Last value of the FBM
    fbm_candidate : float
        Candidate value of the FBM
    space_lim : np.ndarray
        Space limit (min, max) for the FBM

    Returns:
    --------
    float
        New value of the FBM
    """
    if fbm_candidate > space_lim[1]:
        return space_lim[1]
    elif fbm_candidate < space_lim[0]:
        return space_lim[0]
    else:
        return fbm_candidate
