# Camera Configuration Help

This section configures the camera model used in the simulation. Each field controls aspects of detection, noise, resolution, and efficiency.

## Fields

### `type`
- **Type**: `String`
- **Options**: `"CMOS"`
- **Description**: Specifies the camera sensor model. Only CMOS is currently supported in the UI. Programmatically you also have access to EMCCD.

### `pixel_count`
- **Type**: `Array of Integers`
- **Shape**: `[width, height]`
- **Description**: Defines the resolution of the sensor in pixels. Affects image dimensions.

### `pixel_detector_size`
- **Type**: `Number`
- **Units**: `micrometers (μm)`
- **Description**: Physical size of a single pixel. Affects spatial resolution and magnification scaling.

### `magnification`
- **Type**: `Integer`
- **Description**: Optical magnification factor applied to the image.

### `dark_current`
- **Type**: `Number`
- **Units**: `electrons/pixel/second`
- **Description**: Rate at which thermal electrons accumulate in each pixel without illumination.

### `readout_noise`
- **Type**: `Number`
- **Units**: `electrons RMS`
- **Description**: Baseline electronic noise added during readout.

### `bit_depth`
- **Type**: `Integer`
- **Description**: Bit depth of the analog-to-digital converter. Controls the dynamic range of intensity.

### `sensitivity`
- **Type**: `Number`
- **Units**: `electrons/ADU`
- **Description**: Defines how many electrons are required to register one analog-digital unit (ADU) in the image.

### `base_adu`
- **Type**: `Integer`
- **Description**: The base offset in ADUs added to all pixel values.

### `binning_size`
- **Type**: `Integer`
- **Description**: Spatial binning size used to combine adjacent pixels. Common values are 1 (no binning), 2 (2x2 binning), etc.

### `quantum_efficiency`
- **Type**: `Array of [wavelength, efficiency] pairs`
- **Units**: `nm`, `[0-1]`
- **Description**: Describes the wavelength-dependent efficiency of the camera. Higher values mean more photons are converted to signal.

