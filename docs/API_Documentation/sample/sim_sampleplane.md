## `sim_sampleplane.py`

This module defines classes for simulating a sample plane, including fluorescent objects, spatial indexing, and time-based tracking.

### Constants

- **`EMPTY_STATE_HISTORY_DICT`**: A placeholder dictionary indicating no transmittance at a given wavelength.

### Classes

#### `FluorescentObject`

Represents a fluorescent object in the sample plane.

- **Attributes**:
  - `object_id`: Unique identifier for the object.
  - `position`: Current position (x, y, z) in μm.
  - `fluorophore`: `Fluorophore` instance defining the object's properties.
  - `position_history`: Dictionary tracking position over time.
  - `state_history`: Dictionary tracking state history over time.

- **Methods**:
  - `__post_init__()`: Initializes position and state history at t=0.

#### `CellSpaceView`

Represents a 3D field of view with bounds in x, y, and z dimensions.

- **Attributes**:
  - `x_bounds`: Tuple defining the x-axis bounds.
  - `y_bounds`: Tuple defining the y-axis bounds.
  - `z_bounds`: Tuple defining the z-axis bounds.

- **Methods**:
  - `__post_init__()`: Validates that bounds have min < max.
  - `contains(position: Tuple[float, float, float]) -> bool`: Checks if a position is within the FOV.
  - `bounds`: Property returning the FOV bounds.

#### `SampleSpace`

Represents the total 3D space available for simulation.

- **Attributes**:
  - `x_max`: Maximum x-coordinate.
  - `y_max`: Maximum y-coordinate.
  - `z_min`: Minimum z-coordinate.
  - `z_max`: Maximum z-coordinate.

- **Methods**:
  - `contains_fov(fov: CellSpaceView) -> bool`: Checks if a FOV fits within the sample space.
  - `contains_position(position: Tuple[float, float, float]) -> bool`: Checks if a position is within the sample space.
  - `bounds`: Property returning the space bounds.

#### `SamplePlane`

Manages the simulation of fluorescent objects within a sample plane.

- **Attributes**:
  - `_space`: `SampleSpace` instance defining the total simulation volume.
  - `_fov`: `CellSpaceView` instance defining the FOV.
  - `_objects`: Dictionary of `FluorescentObject` instances.
  - `_spatial_index`: Spatial index for fast lookup of objects.
  - `_grid_size`: Size of grid cells in μm.
  - `dt`: Time step in milliseconds.
  - `dt_s`: Time step in seconds.
  - `t_end`: End time in milliseconds.
  - `time_points`: List of time points for simulation.

- **Methods**:
  - `__init__(sample_space: SampleSpace, fov: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], oversample_motion_time: int, t_end: int)`: Initializes the sample plane.
  - `add_object(object_id: str, position: Tuple[float, float, float], fluorophore: Fluorophore, trajectory: Optional[Dict[int, Tuple[float, float, float]]] = None) -> bool`: Adds a fluorescent object to the sample plane.
  - `get_all_objects() -> List[FluorescentObject]`: Retrieves all objects in the sample plane.
  - `get_objects_in_range(center: Tuple[float, float, float], radius: float, t: int) -> List[FluorescentObject]`: Retrieves objects within a specified radius of a point at a given time.
  - `is_within_fov(position: Tuple[float, float, float]) -> bool`: Checks if a position is within the FOV.
  - `_update_spatial_index(object_id: str, position: Tuple[float, float, float], t: int)`: Updates the spatial index for an object at a specific time.
  - `get_object(object_id: str) -> Optional[FluorescentObject]`: Retrieves an object by its ID.
  - `__iter__() -> Iterator[FluorescentObject]`: Makes the sample plane iterable.
  - `__len__() -> int`: Returns the number of objects in the sample plane.
  - `_get_grid_position(position: Tuple[float, float, float]) -> Tuple[int, int, int]`: Converts a position to grid coordinates.
  - `_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float`: Calculates the Euclidean distance between two points.
  - `fov_bounds`: Property returning the FOV bounds.
  - `sample_space_bounds`: Property returning the sample space bounds.