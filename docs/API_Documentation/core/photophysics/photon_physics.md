
# Module Documentation: `photon_physics.py`

## Overview

The `photon_physics.py` module provides classes and methods for simulating photon absorption, emission, and detection processes. It includes models for absorption, emission, and incident photon calculations, as well as methods for handling quantum efficiency and PSF (Point Spread Function) effects.

## Classes

### `AbsorptionBase`

A base class for calculating photon absorption processes.

#### Attributes:
- `excitation_spectrum` (`SpectralData`): The excitation spectrum of the fluorophore.
- `intensity_incident` (`WavelengthDependentProperty`): The incident light intensity.
- `absorb_cross_section_spectrum` (`WavelengthDependentProperty`): The absorption cross-section spectrum.

#### Methods:
- `_calc_flux_density_precursor(self) -> WavelengthDependentProperty`:
  Calculates the flux density precursor for absorption.

---

### `AbsorptionPhysics`

Extends `AbsorptionBase` to include fluorescent lifetime and saturation rate calculations.

#### Attributes:
- `fluorescent_lifetime_inverse` (`float`): The inverse of the fluorescent lifetime.

#### Methods:
- `saturation_rate(self, rate: float, max_rate: float) -> float`:
  Clips the rate to a maximum value.

- `absorbed_photon_rate(self) -> float`:
  Calculates the rate of absorbed photons.

---

### `PhotoStateSwitchPhysics`

Extends `AbsorptionBase` for photostate switching physics.

#### Attributes:
- `quantum_yeild` (`float`): The quantum yield for photostate switching.

---

### `EmissionPhysics`

A class for calculating photon emission rates and transmission.

#### Attributes:
- `emission_spectrum` (`SpectralData`): The emission spectrum of the fluorophore.
- `quantum_yield` (`WavelengthDependentProperty`): The quantum yield for emission.
- `transmission_filter` (`FilterSpectrum`): The transmission filter spectrum.

#### Methods:
- `emission_photon_rate(self, total_absorbed_rate: float) -> WavelengthDependentProperty`:
  Calculates the rate of emitted photons.

- `transmission_photon_rate(self, emission_photon_rate_lambda: WavelengthDependentProperty) -> WavelengthDependentProperty`:
  Calculates the rate of transmitted photons.

---

### `incident_photons`

A class for calculating incident photons based on transmission rates, quantum efficiency, and PSF.

#### Attributes:
- `transmission_photon_rate` (`WavelengthDependentProperty`): The rate of transmitted photons.
- `quantumEff` (`QuantumEfficiency`): The quantum efficiency of the detector.
- `psf` (`Callable`): A function to generate the PSF.
- `position` (`Tuple[float, float, float]`): The position of the fluorophore.

#### Methods:
- `incident_photons_calc(self, dt: float) -> Tuple[float, List]`:
  Calculates the number of incident photons and their distribution.

---

## Dependencies

- `numpy`: For numerical operations.
- `pydantic`: For data validation and settings management.
- `..optics.camera.detectors`: For photon noise calculations.
- `..optics.camera.quantum_eff`: For quantum efficiency calculations.
- `..optics.filters.filters`: For filter spectrum calculations.
- `..optics.psf.psf_engine`: For PSF calculations.
- `..sample.flurophores.flurophore_schema`: For fluorophore schema definitions.
- `..utils.constants`: For physical constants.

---