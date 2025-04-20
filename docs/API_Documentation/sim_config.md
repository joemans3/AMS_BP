# Simulation Configuration File Documentation

This document provides a detailed explanation of the TOML configuration file used for setting up a simulation. The configuration file is structured into several sections, each defining specific parameters for the simulation. Below is a breakdown of each section and its parameters.

---

## 1. **General Configuration**

### `version`
- **Type**: String
- **Pattern**: `^\d+\.\d+$`
- **Description**: Specifies the version of the configuration file format.

### `length_unit`
- **Type**: String
- **Enum**: `["um"]`
- **Description**: The unit of length used throughout the simulation. Always set to `"um"`.

### `time_unit`
- **Type**: String
- **Enum**: `["ms"]`
- **Description**: The unit of time used throughout the simulation. Always set to `"ms"`.

### `diffusion_unit`
- **Type**: String
- **Enum**: `["um^2/s"]`
- **Description**: The unit of diffusion coefficient used throughout the simulation. Always set to `"um^2/s"`.

---

## 2. **Cell Parameters**

### `cell_type`
- **Type**: Union[str, CellType]
- **Description**: Defines the type of the cell to simulate.
  - Supported: RectangularCell, SphericalCell, OvoidCell, RodCell, BuddingCell

### `params`
- **Type**: dictionary of parameter names (String) and values (Any)
- **Description**: Values for the cell_type specified above. The following the general structure:

```python
# SphericalCell
params = {
    "center": [0, 0, 0],   # 3D center coordinates
    "radius": 10.0         # Radius of sphere
}
```


```python
# RodCell
params = {
    "center": [0, 0, 0],       # 3D center coordinates
    "direction": [0, 0, 1],    # Direction vector (will be normalized)
    "height": 20.0,            # Length of the rod
    "radius": 5.0              # Radius of the rod
}
```


```python
# RectangularCell
params = {
    "bounds": [-10, 10, -10, 10, -10, 10]  # [xmin, xmax, ymin, ymax, zmin, zmax]
}
```


```python
# OvoidCell
params = {
    "center": [0, 0, 0],       # 3D center coordinates
    "xradius": 10.0,           # Radius in x-direction
    "yradius": 15.0,           # Radius in y-direction
    "zradius": 20.0            # Radius in z-direction
}
```


---

## 3. **Molecule Parameters**

### `num_molecules`
- **Type**: Array of Integers
- **Description**: Specifies the number of molecules for each type. The length of this array determines the number of molecule types.

### `track_type`
- **Type**: Array of Strings
- **Enum**: `["constant", "fbm"]`
- **Description**: Defines the type of trajectory for each molecule type.

### `diffusion_coefficient`
- **Type**: Array of Arrays of Numbers
- **Description**: Specifies the diffusion coefficients for each molecule type in `um^2/s`.

### `diffusion_track_amount`
- **Type**: Array of Arrays of Numbers
- **Description**: Defines the initial distribution of diffusion coefficients for trajectories.

### `hurst_exponent`
- **Type**: Array of Arrays of Numbers
- **Description**: Specifies the Hurst exponent for each molecule type.

### `hurst_track_amount`
- **Type**: Array of Arrays of Numbers
- **Description**: Defines the initial distribution of Hurst exponents for trajectories.

### `allow_transition_probability`
- **Type**: Array of Booleans
- **Description**: Determines whether transition probabilities are allowed for each molecule type.

### `transition_matrix_time_step`
- **Type**: Array of Integers
- **Description**: The time step in milliseconds (`ms`) at which transition probabilities are applied.

### `diffusion_transition_matrix`
- **Type**: Array of Arrays of Arrays of Numbers
- **Description**: The transition matrix for diffusion coefficients.

### `hurst_transition_matrix`
- **Type**: Array of Arrays of Arrays of Numbers
- **Description**: The transition matrix for Hurst exponents.

### `state_probability_diffusion`
- **Type**: Array of Arrays of Numbers
- **Description**: The probability distribution for diffusion states.

### `state_probability_hurst`
- **Type**: Array of Arrays of Numbers
- **Description**: The probability distribution for Hurst states.

---

## 4. **Global Parameters**

### `sample_plane_dim`
- **Type**: Array of Numbers
- **Shape**: `[width, height]`
- **Description**: The dimensions of the sample plane in micrometers (`um`).

### `cycle_count`
- **Type**: Integer
- **Description**: The number of cycles of exposure and interval time.

### `exposure_time`
- **Type**: Integer
- **Description**: The exposure time in milliseconds (`ms`).

### `interval_time`
- **Type**: Integer
- **Description**: The interval time in milliseconds (`ms`).

### `oversample_motion_time`
- **Type**: Integer
- **Description**: The oversampling time for motion in milliseconds (`ms`).

---

## 5. **Condensate Parameters**

### `initial_centers`
- **Type**: Array of Array of Arrays of Numbers
- **Shape**: `[[[x, y, z], ...], ...]`
- **Description**: The initial centers of condensates in micrometers (`um`). The first dimension is the types of different molecules. Similarr to the `num_molecules` array in the molecule parameters.

### `initial_scale`
- **Type**: Array of Array of Numbers
- **Description**: The initial scale of condensates in micrometers (`um`).

### `diffusion_coefficient`
- **Type**: Array of Array of Numbers
- **Description**: The diffusion coefficients of condensates in `um^2/s`.

### `hurst_exponent`
- **Type**: Array of Array of Numbers
- **Description**: The Hurst exponents of condensates.

### `density_dif`
- **Type**: Array of Number
- **Description**: The density difference between the condensate and the background.

---

## 6. **Output Parameters**

### `output_path`
- **Type**: String
- **Description**: The path where the output files will be saved. Can be absolute or relative.

### `output_name`
- **Type**: String
- **Description**: The name of the output file.

### `subsegment_type`
- **Type**: String
- **Description**: The type of subsegmentation to be applied. Currently not implemented.

### `subsegment_number`
- **Type**: Integer
- **Description**: The number of subsegments. Currently not implemented.

---

## 7. **Fluorophores**

### `num_of_fluorophores`
- **Type**: Integer
- **Description**: The number of fluorophores in the simulation.

### `fluorophore_names`
- **Type**: Array of Strings
- **Description**: The names of the fluorophores.

### **Fluorophore-Specific Parameters**
Each fluorophore has its own section with the following parameters:

#### `name`
- **Type**: String
- **Description**: The name of the fluorophore.

#### `initial_state`
- **Type**: String
- **Condition**: One of the states defined in the `states` section.
- **Description**: The initial state of the fluorophore. 

#### **States**
Each fluorophore has states defined with the following parameters:

- **`name`**: The name of the state.
- **`state_type`**: The type of state (`fluorescent`, `dark`, `bleached`).
- **`quantum_yield`**: The quantum yield of the state (0-1).
- **`extinction_coefficient`**: The extinction coefficient in `M^-1 cm^-1`.
- **`fluorescent_lifetime`**: The fluorescent lifetime in seconds.
- **`excitation_spectrum`**: The excitation spectrum with `wavelengths` and `intensities`.
- **`emission_spectrum`**: The emission spectrum with `wavelengths` and `intensities`.

#### **Transitions**
Each fluorophore has transitions defined with the following parameters:

- **`from_state`**: The starting state of the transition.
- **`to_state`**: The target state of the transition.
- **`photon_dependent`**: Whether the transition is photon-dependent.
- **`spectrum`**: The spectrum associated with the transition, including `wavelengths`, `intensities`, `extinction_coefficient`, and `quantum_yield`.
- **`base_rate`**: The base rate of the transition in `1/s`.

---

## 8. **PSF (Point Spread Function)**

### `type`
- **Type**: String
- **Enum**: `["gaussian"]`
- **Description**: The type of PSF.

### `custom_path`
- **Type**: String
- **Description**: The path to a custom PSF file. Currently not supported.

### `parameters`
- **Type**: Object
- **Properties**:
  - **`numerical_aperture`**: The numerical aperture (typical range: 0.1 - 1.5).
  - **`refractive_index`**: The refractive index (default is air: 1.0).
  - **`pinhole_diameter`**: None or float. In units of um. None = no pinhole before the detector.

---

## 9. **Lasers**

### `active`
- **Type**: Array of Strings
- **Description**: The list of active lasers.

### **Laser-Specific Parameters**
Each laser has its own section with the following parameters:

#### `type`
- **Type**: String
- **Enum**: `["widefield", "gaussian", "hilo"]`
- **Description**: The type of laser.

#### `preset`
- **Type**: String
- **Description**: The preset name of the laser.

#### `parameters`
- **Type**: Object
- **Properties**:
  - **`power`**: The power of the laser in watts.
  - **`wavelength`**: The wavelength in nanometers.
  - **`beam_width`**: The beam width in micrometers.
  - **`numerical_aperture`**: The numerical aperture.
  - **`refractive_index`**: The refractive index.
  - **`inclination_angle`**: The inclination angle in degrees (only for HiLo).

---

## 10. **Channels**

### `num_of_channels`
- **Type**: Integer
- **Description**: The number of channels.

### `channel_names`
- **Type**: Array of Strings
- **Description**: The names of the channels.

### `split_efficiency`
- **Type**: Array of Numbers
- **Description**: The efficiency of the channel splitter for each channel (0-1).

### **Filter Set Configuration**
Each channel has its own filter set with the following parameters:

#### `filter_set_name`
- **Type**: String
- **Description**: The name of the filter set.

#### `filter_set_description`
- **Type**: String
- **Description**: The description of the filter set.

#### `excitation`
- **Type**: Object
- **Properties**:
  - **`name`**: The name of the excitation filter.
  - **`type`**: The type of filter (`allow_all`).
  - **`points`**: The number of points in the filter.

#### `emission`
- **Type**: Object
- **Properties**:
  - **`name`**: The name of the emission filter.
  - **`type`**: The type of filter (`bandpass`).
  - **`center_wavelength`**: The center wavelength in nanometers.
  - **`bandwidth`**: The bandwidth in nanometers.
  - **`transmission_peak`**: The transmission peak (0-1).
  - **`points`**: The number of points in the filter.

---

## 11. **Camera**

### `type`
- **Type**: String
- **Enum**: `["CMOS"]`
- **Description**: The type of camera.

### `pixel_count`
- **Type**: Array of Integers
- **Shape**: `[width, height]`
- **Description**: The number of pixels in the camera.

### `pixel_detector_size`
- **Type**: Number
- **Description**: The size of each pixel detector in micrometers.

### `magnification`
- **Type**: Integer
- **Description**: The magnification of the camera.

### `dark_current`
- **Type**: Number
- **Description**: The dark current in electrons per pixel per second.

### `readout_noise`
- **Type**: Number
- **Description**: The readout noise in electrons RMS.

### `bit_depth`
- **Type**: Integer
- **Description**: The bit depth of the camera.

### `sensitivity`
- **Type**: Number
- **Description**: The sensitivity of the camera in electrons per ADU.

### `base_adu`
- **Type**: Integer
- **Description**: The base ADU value.

### `binning_size`
- **Type**: Integer
- **Description**: The binning size (e.g., 1x1 or 2x2).

### `quantum_efficiency`
- **Type**: Array of Arrays of Numbers
- **Shape**: `[[wavelength, efficiency], ...]`
- **Description**: The quantum efficiency curve of the camera.

---

## 12. **Experiment**

### `name`
- **Type**: String
- **Description**: The name of the experiment.

### `description`
- **Type**: String
- **Description**: The description of the experiment.

### `experiment_type`
- **Type**: String
- **Enum**: `["time-series", "z-stack"]`
- **Description**: The type of experiment.

### `z_position`
- **Type**: Array of Numbers, or Number ( if "z-stack" or "time-series")
- **Description**: The z-positions for the experiment.

### `laser_names_active`
- **Type**: Array of Strings
- **Description**: The names of active lasers.

### `laser_powers_active`
- **Type**: Array of Numbers
- **Description**: The powers of active lasers in watts.

### `laser_positions_active`
- **Type**: Array of Arrays of Numbers
- **Shape**: `[[x, y, z], ...]`
- **Description**: The positions of active lasers in micrometers.

### `xyoffset`
- **Type**: Array of Numbers
- **Shape**: `[x, y]`
- **Description**: The x and y offsets in micrometers.

### `exposure_time`
- **Type**: Integer
- **Description**: The exposure time in milliseconds.

### `interval_time`
- **Type**: Integer
- **Description**: The interval time in milliseconds.

---
