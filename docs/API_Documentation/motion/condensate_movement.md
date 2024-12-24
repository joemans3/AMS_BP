# Module: condensate_movement.py

## Overview

The `condensate_movement` module contains classes and functions for simulating and managing the movement and properties of condensates. Condensates are defined as spherical entities characterized by their center coordinates (x, y, z), radius (r), and time (t). This module provides functionalities to create, manipulate, and analyze condensates within a simulation environment.

## Classes

### `Condensate`

#### Description

The `Condensate` class is used to store and manage data related to condensates. Each condensate is defined by its initial position, time, diffusion coefficient, Hurst exponent, and other relevant parameters.

#### Parameters

- **initial_position**: `np.ndarray`  
  The initial position of the condensate. Default is `np.array([0, 0, 0])`.

- **initial_time**: `float`  
  The initial time of the condensate. Default is `0`.

- **diffusion_coefficient**: `float`  
  The diffusion coefficient of the condensate. Default is `0`.

- **hurst_exponent**: `float`  
  The Hurst exponent of the condensate. Default is `0`.

- **units_time**: `str`  
  The units of time. Default is `'s'`. The time units work as follows:
  - For `'ms'`, 1 unit represents 1 millisecond.
  - For `'s'`, 1 unit represents 1 second.
  - For `'20ms'`, 1 unit represents 20 milliseconds.

- **units_position**: `str`  
  The units of position. Default is `'um'`.

- **condensate_id**: `int`  
  The ID of the condensate. Default is `0`.

- **initial_scale**: `float`  
  The initial scale of the condensate. Default is `0`.

#### Usage

```python
condensate = Condensate(**{
    "initial_position": np.array([0, 0, 0]),
    "initial_time": 0,
    "diffusion_coefficient": 0,
    "hurst_exponent": 0,
    "units_time": 'ms',
    "units_position": 'um',
    "condensate_id": 0,
    "initial_scale": 0,
})
```

## Functions

### `create_condensate_dict`

#### Description

Creates a dictionary of condensates for simulation. Each condensate is initialized with specific parameters such as initial centers, scales, diffusion coefficients, and Hurst exponents.

#### Parameters

- **initial_centers**: `np.ndarray`  
  An array of shape `(num_condensates, 2)` representing the initial centers of the condensates.

- **initial_scale**: `np.ndarray`  
  An array of shape `(num_condensates, 2)` representing the initial scales of the condensates.

- **diffusion_coefficient**: `np.ndarray`  
  An array of shape `(num_condensates, 2)` representing the diffusion coefficients of the condensates.

- **hurst_exponent**: `np.ndarray`  
  An array of shape `(num_condensates, 2)` representing the Hurst exponents of the condensates.

- **cell**: `RectangularCell`  
  The rectangular cell that contains the condensates.

- **kwargs**: `dict`  
  Additional arguments passed to the `Condensate` class.

#### Returns

- **dict**: A dictionary of `Condensate` objects with keys as condensate IDs.

#### Usage

```python
condensates = create_condensate_dict(
    initial_centers=np.array([[0, 0], [1, 1]]),
    initial_scale=np.array([1, 2]),
    diffusion_coefficient=np.array([0.1, 0.2]),
    hurst_exponent=np.array([0.5, 0.6]),
    cell=RectangularCell(),
)
```

### `_generate_transition_tracks`

#### Description

Generates tracks with state transitions for condensates. This function is used to simulate the movement of condensates under varying conditions defined by transition matrices and state probabilities.

#### Parameters

- **track_generator**: `Track_generator`  
  An instance of the track generator.

- **track_lengths**: `list | np.ndarray | int`  
  The lengths of the tracks to be generated.

- **initial_positions**: `np.ndarray`  
  The initial positions of the condensates.

- **starting_frames**: `int`  
  The starting frames for the tracks.

- **diffusion_parameters**: `np.ndarray`  
  The diffusion parameters for the condensates.

- **hurst_parameters**: `np.ndarray`  
  The Hurst parameters for the condensates.

- **diffusion_transition_matrix**: `np.ndarray`  
  The transition matrix for diffusion states.

- **hurst_transition_matrix**: `np.ndarray`  
  The transition matrix for Hurst states.

- **diffusion_state_probability**: `np.ndarray`  
  The state probability for diffusion.

- **hurst_state_probability**: `np.ndarray`  
  The state probability for Hurst.

#### Returns

- **tuple[dict, dict]**: A tuple containing the tracks dictionary and points per time dictionary.

### `_convert_tracks_to_trajectory`

#### Description

Converts the generated tracks into a trajectory format.

#### Parameters

- **tracks**: `dict`  
  The dictionary of tracks to be converted.

#### Returns

- **dict**: The converted trajectory data.

## Dependencies

- `matplotlib.pyplot`
- `numpy`
- `RectangularCell` from `..cells.rectangular_cell`
- `cache` from `..utils.decorators`
- `Track_generator` from `track_gen`

## Usage Example

```python
import numpy as np
from ..cells.rectangular_cell import RectangularCell
from .condensate_movement import create_condensate_dict, Condensate

# Initialize a rectangular cell
cell = RectangularCell()

# Create condensates
condensates = create_condensate_dict(
    initial_centers=np.array([[0, 0], [1, 1]]),
    initial_scale=np.array([1, 2]),
    diffusion_coefficient=np.array([0.1, 0.2]),
    hurst_exponent=np.array([0.5, 0.6]),
    cell=cell,
)

# Access a condensate
condensate = condensates['0']

# Get position and scale at a given time
result = condensate(times=10, time_unit='ms')
print(result)
```

This documentation provides a comprehensive overview of the `condensate_movement` module, detailing its classes, functions, and dependencies. The provided example demonstrates how to initialize and use the `Condensate` class and related functions for simulating condensate movements.