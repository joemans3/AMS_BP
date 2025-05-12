from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ...utils.constants import H_C_COM
from ..optics.camera.detectors import photon_noise
from ..optics.camera.quantum_eff import QuantumEfficiency
from ..optics.filters.filters import FilterSpectrum
from ..optics.psf.psf_engine import PSFEngine
from ..sample.flurophores.flurophore_schema import (
    SpectralData,
    WavelengthDependentProperty,
)


@dataclass
class AbsorptionBase:
    excitation_spectrum: SpectralData  # wl in nm, relative intensity
    intensity_incident: WavelengthDependentProperty  # wl in nm, intensity in W/um^2
    absorb_cross_section_spectrum: (
        WavelengthDependentProperty  # wl in nm, cross section in cm^2
    )

    def __post_init__(self):
        self.flux_density_precursor_lambda = self._calc_flux_density_precursor()

    def _calc_flux_density_precursor(self) -> WavelengthDependentProperty:
        """Per wavelength of incoming light W/cm^2 (intensity), find the quantity W/cm over the excitation_spectrum provided"""
        wavelengths = []
        ex_flux_density_lambda = []
        for i in range(len(self.intensity_incident.wavelengths)):
            intensity = self.intensity_incident.values[i]
            wavelength = self.intensity_incident.wavelengths[i]
            ex_spectrum = self.excitation_spectrum.get_value(wavelength)
            ex_flux_density_lambda.append(intensity * wavelength * ex_spectrum)
            wavelengths.append(wavelength)
        return WavelengthDependentProperty(
            wavelengths=wavelengths, values=ex_flux_density_lambda
        )


@dataclass
class AbsorptionPhysics(AbsorptionBase):
    fluorescent_lifetime_inverse: float

    def saturation_rate(self, rate: float, max_rate: float) -> float:
        return np.clip(rate, 0, max_rate)

    def absorbed_photon_rate(self) -> float:
        """Calculate the rate of incident photons"""
        if self.flux_density_precursor_lambda is None:
            raise ValueError("Flux density not calculated")

        photon_rate_lambda = 0  # adding up all the wavelength based intensity rates
        for i in range(len(self.flux_density_precursor_lambda.wavelengths)):
            cross_section = self.absorb_cross_section_spectrum.values[i]
            int_inverse_seconds_i = (
                cross_section
                * self.flux_density_precursor_lambda.values[i]
                * H_C_COM
                * 1e-1
            )

            photon_rate_lambda += int_inverse_seconds_i
        return self.saturation_rate(
            photon_rate_lambda,
            self.fluorescent_lifetime_inverse,
        )  # 1/s, 10^-1 combined all conversion factors


@dataclass
class PhotoStateSwitchPhysics(AbsorptionBase):
    quantum_yeild: float  # switching events per photon absorbed


@dataclass
class EmissionPhysics:
    emission_spectrum: SpectralData  # wl in nm, normalied intensity
    quantum_yield: WavelengthDependentProperty
    transmission_filter: FilterSpectrum

    def __post_init__(self):
        # normalize emission spectrum
        emission_spectrum_sum = sum(self.emission_spectrum.values)
        self.emission_spectrum = SpectralData(
            wavelengths=self.emission_spectrum.wavelengths,
            intensities=[
                val / emission_spectrum_sum for val in self.emission_spectrum.values
            ],
        )

    def emission_photon_rate(
        self,
        total_absorbed_rate: float,  # 1/s
    ) -> WavelengthDependentProperty:
        """Calculate the rate of emitted photons (1/s)

        Parameters:
            total_absorbed_rate: float
        """

        wavelengths = []
        emission_rate_lambda = []
        for i in range(len(self.emission_spectrum.wavelengths)):
            wavelengths.append(self.emission_spectrum.wavelengths[i])
            emission_rate_lambda.append(
                total_absorbed_rate
                * self.quantum_yield.values[i]
                * self.emission_spectrum.values[i]
            )

        return WavelengthDependentProperty(
            wavelengths=wavelengths, values=emission_rate_lambda
        )

    def transmission_photon_rate(
        self, emission_photon_rate_lambda: WavelengthDependentProperty
    ) -> WavelengthDependentProperty:
        """Calculate the rate of transmitted photons (1/s)

        Parameters:
            emission_photon_rate_lambda: WavelengthDependentProperty
        """
        wavelengths = []
        transmission_rate_lambda = []
        for i in range(len(emission_photon_rate_lambda.wavelengths)):
            wavelengths.append(emission_photon_rate_lambda.wavelengths[i])
            transmission_rate_lambda.append(
                emission_photon_rate_lambda.values[i]
                * self.transmission_filter.find_transmission(
                    emission_photon_rate_lambda.wavelengths[i]
                )
            )

        return WavelengthDependentProperty(
            wavelengths=wavelengths, values=transmission_rate_lambda
        )


@dataclass
class incident_photons:
    transmission_photon_rate: WavelengthDependentProperty
    quantumEff: QuantumEfficiency
    psf: Callable[[float | int, Optional[float | int]], PSFEngine]
    position: Tuple[float, float, float]

    def __post_init__(self):
        self.generator = []
        for i in range(len(self.transmission_photon_rate.wavelengths)):
            if self.transmission_photon_rate.values[i] > 0:
                self.generator.append(
                    self.psf(
                        self.transmission_photon_rate.wavelengths[i], self.position[2]
                    )
                )
            else:
                self.generator.append(0)

    def incident_photons_calc(
        self, dt: float, collection_efficiency: float = 1
    ) -> Tuple[float, List]:
        photons = 0
        psf_hold = []
        for i in range(len(self.transmission_photon_rate.wavelengths)):
            if self.transmission_photon_rate.values[i] > 0:
                qe_lam = self.quantumEff.get_qe(
                    self.transmission_photon_rate.wavelengths[i]
                )
                photons_n = (
                    self.transmission_photon_rate.values[i] * dt * collection_efficiency
                )
                photons += photons_n
                psf_gen = (
                    self.generator[i].psf_z(
                        x_val=self.position[0],
                        y_val=self.position[1],
                        z_val=self.position[2],
                    )
                    * photons_n
                )

                psf_hold.append(photon_noise(psf_gen) * qe_lam)

        return photons, psf_hold
