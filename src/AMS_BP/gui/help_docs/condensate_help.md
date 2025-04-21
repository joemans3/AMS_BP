# Condensate Configuration Help

This section defines parameters for **condensate droplets** within the simulation. Each tab corresponds to a molecule type, and each type can have multiple condensates.

---

## Molecule Count

### `Number of Molecule Types`
- **Type**: Integer
- **Description**: The number of different molecule types in the simulation. Each will be assigned its own tab.

---

## Per-Molecule Condensate Parameters

### `Number of Condensates`
- **Type**: Integer
- **Description**: The number of condensates of this molecule type.

Each condensate defines the following:

### `Initial Center`
- **Type**: List of 3 Floats `[x, y, z]`
- **Unit**: Micrometers (µm)
- **Description**: The spatial center of the condensate in 3D.

### `Initial Scale`
- **Type**: Float
- **Unit**: Micrometers (µm)
- **Description**: The initial physical size (radius or diameter) of the condensate.

### `Diffusion Coefficient`
- **Type**: Float
- **Unit**: µm²/s
- **Description**: The diffusion coefficient for the condensate particles.

### `Hurst Exponent`
- **Type**: Float
- **Range**: `0.0` – `1.0`
- **Description**: Controls the memory of the random walk. `0.5` = standard Brownian motion. Values above or below indicate persistent or antipersistent motion.

---

## Density Difference

### `Density Difference`
- **Type**: Float
- **Description**: Represents the relative density difference between the condensate and its surrounding medium.
- **Location**: Set **per molecule type** (i.e., one value per tab).

---

### Notes

- All condensate values are grouped per molecule type.
- You may define different numbers of condensates for different molecule types.
- This module supports nested array generation for initialization within the simulation core.

