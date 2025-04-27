
# Documentation for `fbm_BP.py`

## Overview

The `fbm_BP.py` module provides a class and utility functions for simulating fractional Brownian motion (FBM) with Markov processes for diffusion coefficients and Hurst exponents. The module also includes functionality to apply boundary conditions to the FBM trajectory.

## Classes

### `FBM_BP`

```python
class FBM_BP:
```

#### Description

Simulates fractional Brownian motion (FBM) using a Markov process for diffusion coefficients and Hurst exponents. The class allows for adjustable parameters and boundary conditions to control the simulation.

#### Parameters

- **n** (`int`): Number of time steps in the simulation.
- **dt** (`float`): Time step duration in milliseconds.
- **diffusion_parameters** (`np.ndarray`): Array of diffusion coefficients for the FBM simulation.
- **hurst_parameters** (`np.ndarray`): Array of Hurst exponents for the FBM simulation.
- **diffusion_parameter_transition_matrix** (`np.ndarray`): Transition matrix for diffusion coefficients.
- **hurst_parameter_transition_matrix** (`np.ndarray`): Transition matrix for Hurst exponents.
- **state_probability_diffusion** (`np.ndarray`): Initial probabilities of different diffusion states.
- **state_probability_hurst** (`np.ndarray`): Initial probabilities of different Hurst states.
- **space_lim** (`np.ndarray`): Space limits (min, max) for the FBM.

#### Methods

- **`_autocovariance(k: int, hurst: float) -> float`**:
  - Computes the autocovariance function for fractional Gaussian noise (fGn).

- **`_setup() -> None`**:
  - Precomputes the autocovariance matrix and sets up initial diffusion and Hurst parameters.

- **`fbm() -> np.ndarray`**:
  - Runs the FBM simulation and returns the positions at each time step.

---

## Functions

### `_boundary_conditions`

```python
def _boundary_conditions(
    fbm_store_last: float,
    fbm_candidate: float,
    space_lim: np.ndarray,
    condition_type: str,
) -> float:
```

#### Description

Applies boundary conditions to the FBM simulation. The function selects the appropriate boundary condition (reflecting or absorbing) based on the `condition_type` parameter.

#### Parameters

- **fbm_store_last** (`float`): The last value of the FBM trajectory.
- **fbm_candidate** (`float`): The candidate value for the next step in the FBM trajectory.
- **space_lim** (`np.ndarray`): A 2-element array representing the minimum and maximum space limits.
- **condition_type** (`str`): The type of boundary condition to apply, either "reflecting" or "absorbing".

#### Returns

- **float**: The new value for the FBM trajectory, adjusted by the specified boundary condition.

#### Raises

- **ValueError**: If the `condition_type` is not one of the supported boundary conditions.

---

## Constants

### `BOUNDARY_CONDITIONS`

```python
BOUNDARY_CONDITIONS = {
    "reflecting": _refecting_boundary,
    "absorbing": _absorbing_boundary,
}
```

#### Description

A dictionary mapping boundary condition types to their corresponding functions. This allows for easy selection of boundary conditions during the simulation.

#### Keys

- **"reflecting"**: Reflecting boundary condition.
- **"absorbing"**: Absorbing boundary condition.