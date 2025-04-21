# Point Spread Function (PSF) Configuration Help

The PSF (Point Spread Function) models how light emitted from a point source is blurred by the microscope optics. This section defines the parameters of the optical model used for image formation.

---

## `type`
- **Type**: String
- **Allowed**: `"gaussian"` *(currently the only supported type)*
- **Description**: Specifies the PSF model. A Gaussian PSF approximates diffraction-limited optics and is sufficient for many simulation scenarios.

---

## `custom_path`
- **Type**: String
- **Default**: Empty
- **Description**: Placeholder for loading a custom PSF file. Currently not supported.

---

## `confocal`
- **Type**: Boolean
- **Description**: Enables confocal mode by introducing a pinhole before the detector to reject out-of-focus light. When enabled, additional parameters become relevant (e.g., pinhole diameter).

---

## `parameters`

### `numerical_aperture`
- **Type**: Float
- **Typical Range**: 0.1 – 1.5
- **Description**: The numerical aperture of the objective lens, which affects the resolution and depth of field of the microscope.

### `refractive_index`
- **Type**: Float
- **Typical Range**: 1.0 – 2.0
- **Description**: The refractive index of the imaging medium (e.g., air: 1.0, water: 1.33, immersion oil: ~1.5).

### `pinhole_diameter`
- **Type**: Float (optional)
- **Units**: Micrometers (μm)
- **Visible only when confocal mode is enabled**
- **Description**: The diameter of the confocal pinhole that restricts the depth of detection, improving axial resolution by eliminating out-of-focus light.

---

## Notes

- Confocal simulation is optional and intended for advanced users modeling high-resolution 3D optical sectioning.
- You can adjust the `numerical_aperture` and `refractive_index` even in widefield mode to simulate different objective lens setups.

