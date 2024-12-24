# Documentation for `boundary_conditions.py`

## Overview

The `boundary_conditions.py` module provides functions to handle boundary conditions for fractional Brownian motion (FBM) simulations. These functions ensure that the FBM trajectory remains within specified space limits by applying either reflecting or absorbing boundary conditions.

## Functions

### `_refecting_boundary`

```python
@_catch_recursion_error
def _refecting_boundary(
    fbm_store_last: float, fbm_candidate: float, space_lim: np.ndarray
) -> float:
```

#### Description

Applies a reflecting boundary condition to the FBM simulation. If the candidate value exceeds the space limits, the function recursively adjusts the value by reflecting it back into the allowed space.

#### Parameters

- **fbm_store_last** (`float`): The last value of the FBM trajectory.
- **fbm_candidate** (`float`): The candidate value for the next step in the FBM trajectory.
- **space_lim** (`np.ndarray`): A 2-element array representing the minimum and maximum space limits.

#### Returns

- **float**: The new value for the FBM trajectory, adjusted by the reflecting boundary condition.

---

### `_absorbing_boundary`

```python
@_catch_recursion_error
def _absorbing_boundary(
    fbm_store_last: float, fbm_candidate: float, space_lim: np.ndarray
) -> float:
```

#### Description

Applies an absorbing boundary condition to the FBM simulation. If the candidate value exceeds the space limits, the function sets the value to the boundary limit.

#### Parameters

- **fbm_store_last** (`float`): The last value of the FBM trajectory.
- **fbm_candidate** (`float`): The candidate value for the next step in the FBM trajectory.
- **space_lim** (`np.ndarray`): A 2-element array representing the minimum and maximum space limits.

#### Returns

- **float**: The new value for the FBM trajectory, adjusted by the absorbing boundary condition.

---