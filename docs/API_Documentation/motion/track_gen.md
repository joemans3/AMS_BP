# `track_gen.py` Documentation

## Overview

The `track_gen.py` module provides functionality to simulate foci dynamics in space, particularly within cell simulations. It includes a `Track_generator` class that generates tracks of foci movements in a simulated cell space, with options for transitions between different diffusion coefficients and Hurst exponents.

## Classes

### `Track_generator`

A class to generate tracks of foci movements in a simulated cell space.

#### Parameters

- **cell** (`CellType`): 
  - A cell object defining the space for track generation.
- **cycle_count** (`int`): 
  - The number of frames for the simulation.
- **exposure_time** (`int | float`): 
  - Exposure time in milliseconds.
- **interval_time** (`int | float`): 
  - Interval time between frames in milliseconds.
- **oversample_motion_time** (`int | float`): 
  - Time for oversampling motion in milliseconds.

#### Methods

- **`__init__(self, cell: CellType, cycle_count: int, exposure_time: int | float, interval_time: int | float, oversample_motion_time: int | float) -> None`**:
  - Initializes the `Track_generator` object.

- **`_allowable_cell_types(self)`**:
  - Checks if the cell type is supported (currently only rectangular cells are supported).

- **`track_generation_no_transition(self, diffusion_coefficient: float, hurst_exponent: float, track_length: int, initials: np.ndarray, start_time: int | float) -> dict`**:
  - Simulates track generation with no transition between diffusion coefficients and Hurst exponents.

- **`track_generation_with_transition(self, diffusion_transition_matrix: np.ndarray | list, hurst_transition_matrix: np.ndarray | list, diffusion_parameters: np.ndarray | list, hurst_parameters: np.ndarray | list, diffusion_state_probability: np.ndarray | list, hurst_state_probability: np.ndarray | list, track_length: int, initials: np.ndarray, start_time: int | float) -> dict`**:
  - Generates track data with transitions between diffusion coefficients and Hurst exponents.

- **`track_generation_constant(self, track_length: int, initials: np.ndarray, start_time: int) -> dict`**:
  - Generates a constant track (no movement).

- **`_convert_diffcoef_um2s_um2xms(self, diffusion_coefficient: float | np.ndarray | list) -> float | np.ndarray | list`**:
  - Converts diffusion coefficient from `um^2/s` to `um^2/x ms`.

- **`_convert_time_to_frame(self, time: int) -> int`**:
  - Converts time in milliseconds to frame number.

- **`_convert_frame_to_time(self, frame: int) -> int`**:
  - Converts frame number to time in milliseconds.

#### Private Methods

- **`_allowable_cell_types(self)`**:
  - Ensures that only rectangular cells are supported for track generation.

## Functions

- **`_initialize_points_per_time(total_time: int, oversample_motion_time: int) -> dict`**:
  - Initializes an empty dictionary with keys for each time point.

- **`_update_points_per_time(points_per_time: dict, track: dict) -> None`**:
  - Updates the points per time dictionary with new track data.

- **`_generate_constant_tracks(track_generator: Track_generator, track_lengths: list | np.ndarray | int, initial_positions: np.ndarray, starting_frames: int = 0) -> tuple[dict, dict]`**:
  - Generates tracks with constant parameters.

- **`_generate_no_transition_tracks(track_generator: Track_generator, track_lengths: list | np.ndarray | int, initial_positions: np.ndarray, starting_frames: int, diffusion_parameters: np.ndarray, hurst_parameters: np.ndarray) -> tuple[dict, dict]`**:
  - Generates tracks without state transitions.

- **`_generate_transition_tracks(track_generator: Track_generator, track_lengths: list | np.ndarray | int, initial_positions: np.ndarray, starting_frames: int, diffusion_parameters: np.ndarray, hurst_parameters: np.ndarray, diffusion_transition_matrix: np.ndarray, hurst_transition_matrix: np.ndarray, diffusion_state_probability: np.ndarray, hurst_state_probability: np.ndarray) -> tuple[dict, dict]`**:
  - Generates tracks with state transitions.

- **`_convert_tracks_to_trajectory(tracks: dict) -> dict`**:
  - Converts tracks to a trajectory format.

## Dependencies

- `random`
- `numpy` (`np`)
- `RectangularCell`, `RodCell`, `SphericalCell` from `..cells`
- `FBM_BP` from `.movement.fbm_BP`

## Usage

To use the `Track_generator` class, instantiate it with a cell object and the necessary parameters. Then, use the provided methods to generate tracks with or without transitions.

```python
from track_gen import Track_generator
from ..cells.rectangular_cell import RectangularCell

# Example usage
cell = RectangularCell()
track_generator = Track_generator(cell, cycle_count=100, exposure_time=10, interval_time=5, oversample_motion_time=2)

# Generate a track with no transitions
track_data = track_generator.track_generation_no_transition(
    diffusion_coefficient=1.0,
    hurst_exponent=0.5,
    track_length=100,
    initials=np.array([0, 0, 0]),
    start_time=0
)

print(track_data)
```

This will output a dictionary containing the generated track data.