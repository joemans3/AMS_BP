# Laser Configuration Help

This section allows you to define the properties of each laser used in the simulation. Each laser must have a unique name and type, and may include physical and optical characteristics that influence excitation and activation.

---

## General Fields

### `active`
- **Type**: Array of Strings
- **Description**: The list of lasers that are active during the experiment. These names must match the defined laser entries.

---

## Laser-Specific Parameters

Each laser has a dedicated section and includes the following fields:

### `name`
- **Type**: String
- **Description**: Unique identifier for the laser. This name is used to associate laser settings with experiment configurations.

---

### `type`
- **Type**: Enum [`widefield`, `gaussian`, `hilo`]
- **Description**:
  - `widefield`: Uniform illumination across the entire plane.
  - `gaussian`: Focused beam with Gaussian profile.
  - `hilo`: Highly inclined and laminated optical sheet—oblique angle illumination for reduced background.

---

### `preset`
- **Type**: String
- **Description**: An optional label to describe this laser's use case (e.g., `"default 488nm"`). Does not affect simulation directly.

---

### `parameters`

#### `power`
- **Type**: Float
- **Units**: Watts (W)
- **Description**: The power of the laser beam. Impacts excitation intensity and photophysics.

#### `wavelength`
- **Type**: Integer
- **Units**: Nanometers (nm)
- **Description**: The wavelength of emitted light from the laser. Should match fluorophore excitation peaks.

#### `beam_width`
- **Type**: Float
- **Units**: Micrometers (`μm`)
- **Description**: The width of the laser beam in the focal plane. Relevant for spatial precision.

#### `numerical_aperture`
- **Type**: Float
- **Description**: The NA of the illumination optics. Impacts the cone angle and axial resolution.

#### `refractive_index`
- **Type**: Float
- **Description**: The refractive index of the medium the laser is passing through (usually water or oil immersion).

#### `inclination_angle`
- **Type**: Float (only for `hilo`)
- **Units**: Degrees
- **Description**: The angle of incidence for HiLo illumination.

---

## Notes

- **Confocal Mode**: If confocal mode is enabled via PSF settings, all lasers are automatically forced to use Gaussian beams.
- The `inclination_angle` is only enabled when `type` is set to `"hilo"`.

