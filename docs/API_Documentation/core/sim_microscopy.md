# VirtualMicroscope Module Documentation

This module defines the `VirtualMicroscope` class and related utility classes and functions for simulating a virtual microscope. The module leverages various components such as cameras, lasers, sample planes, and photophysics to simulate imaging processes.

## Table of Contents

- [VirtualMicroscope](#virtualmicroscope)

- [mapSampleCamera](#mapsamplecamera)

- [PhotonFrameContainer](#photonframecontainer)

- [Utility Functions](#utility-functions)
  - [generate_sampling_pattern](#generate_sampling_pattern)
  - [timeValidator](#timevalidator)

---

## VirtualMicroscope

The `VirtualMicroscope` class represents a virtual microscope that integrates various components to simulate imaging processes.

### Attributes

- **camera**: Tuple containing a `Detector` and `QuantumEfficiency` object.
- **qe**: Quantum efficiency of the camera.
- **sample_plane**: A `SamplePlane` object representing the sample being imaged.
- **lasers**: Dictionary of `LaserProfile` objects, keyed by laser names.
- **channels**: A `Channels` object representing the imaging channels.
- **psf**: A callable that generates a `PSFEngine` based on input parameters.
- **_time**: Current simulation time in milliseconds.
- **config**: A `ConfigList` object containing configuration parameters.
- **initial_config**: Cached initial configuration of the microscope.

### Methods

#### `__init__(self, camera: Tuple[Detector, QuantumEfficiency], sample_plane: SamplePlane, lasers: Dict[str, LaserProfile], channels: Channels, psf: Callable[[float | int, Optional[float | int]], PSFEngine], config: ConfigList, start_time: int = 0)`

Initializes the `VirtualMicroscope` with the provided components.

#### `_set_laser_position_center_cell(self) -> None`

Centers the laser positions at the center of the sample plane.

#### `_cached_initial_config(self) -> None`

Caches the initial configuration of the microscope.

#### `_set_laser_powers(self, laser_power: Dict[str, float]) -> None`

Sets the power of the lasers. Raises a `ValueError` if the provided power exceeds the maximum allowed power.

#### `_set_laser_positions(self, laser_positions: Dict[str, Tuple[float, float, float]]) -> None`

Sets the positions of the lasers.

#### `run_sim(self, z_val: float, laser_power:Dict[str, Union[float, Callable[[float], float]], xyoffset: Tuple[float, float], laser_position: Optional[Dict[str, Union[Tuple[float, float, float], Callable[[float], Tuple[float, float, float]]]]] = None, duration_total: Optional[int] = None, exposure_time: Optional[int] = None, interval_time: Optional[int] = None, scanning: Optional[bool] = False) -> Tuple[np.ndarray, MetaData]`

Runs the simulation for the given parameters and returns the resulting image stack and metadata.

#### `reset_to_initial_config(self) -> bool`

Resets the microscope to its initial configuration.

---

## mapSampleCamera

The `mapSampleCamera` class maps the location of the sample plane onto the detector grid.

### Attributes

- **sampleplane**: A `SamplePlane` object.
- **camera**: A `Detector` object.
- **xyoffset**: Tuple representing the offset in micrometers.
- **frames**: Number of frames to process.

### Methods

#### `__post_init__(self)`

Initializes the `PhotonFrameContainer` with base frames.

#### `get_pixel_indices(self, x: float, y: float) -> Tuple[int, int]`

Converts sample plane coordinates to detector grid indices.

#### `add_psf_frame(self, psf: np.ndarray, mol_pos: Tuple[float, float], frame_num: int) -> None`

Adds a PSF to the specified frame at the given molecular position.

#### `get_frame(self, frame_num: int) -> np.ndarray`

Returns the specified frame.

---

## PhotonFrameContainer

The `PhotonFrameContainer` class is a container for simulation frames.

### Attributes

- **frames**: List of `np.ndarray` representing the frames.

### Methods

#### `__iter__(self)`

Allows iteration over the frames.

#### `__len__(self)`

Returns the number of frames.

---

## Utility Functions

### generate_sampling_pattern

Generates a sampling pattern based on exposure and interval times.

#### Parameters

- **exposure_time**: Duration of each exposure.
- **interval_time**: Duration between exposures.
- **start_time**: Beginning of the sampling period.
- **end_time**: End of the sampling period.
- **oversample_motion_time**: Time resolution for oversampling.

#### Returns

- **times**: List of sampling times.
- **sample_bool**: List indicating frame numbers or intervals.
- **max_frame**: Maximum frame number.

### timeValidator

Validates the provided time parameters against the simulation constraints.

#### Parameters

- **oexposure_time**: Original exposure time.
- **ointerval_time**: Original interval time.
- **oversample_motion_time**: Time resolution for oversampling.
- **ototal_time**: Original total time.
- **current_time**: Current simulation time.
- **state_arr**: Dictionary containing exposure, interval, and total time.

#### Returns

- **duration_total**: Validated total duration.
- **exposure_time**: Validated exposure time.
- **interval_time**: Validated interval time.

#### Raises

- `ValueError`: If the provided times are invalid.

---
