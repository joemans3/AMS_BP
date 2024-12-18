from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from ..optics.camera.detectors import photon_noise
from ..optics.camera.quantum_eff import QuantumEfficiency
from ..optics.filters.filters import FilterSpectrum
from ..optics.psf.psf_engine import PSFEngine
from ..sample.flurophores.flurophore_schema import (
    SpectralData,
    WavelengthDependentProperty,
)
from ..utils.constants import H_C_COM


@dataclass
class AbsorptionPhysics:
    excitation_spectrum: SpectralData  # wl in nm, relative intensity
    intensity_incident: WavelengthDependentProperty  # wl in nm, intensity in W/um^2
    absorb_cross_section_spectrum: (
        WavelengthDependentProperty  # wl in nm, cross section in cm^2
    )
    fluorescent_lifetime: float  # in 1/s
    flux_density_lambda: Optional[WavelengthDependentProperty] = None

    def __post_init__(self):
        if self.flux_density_lambda is None:
            self.flux_density_lambda = self._calc_flux_density()

    def _calc_flux_density(self) -> WavelengthDependentProperty:
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

    def absorbed_photon_rate(self) -> float:
        """Calculate the number of incident photons"""
        if self.flux_density_lambda is None:
            raise ValueError("Flux density not calculated")

        # calculate number of photons
        photon_rate_lambda = 0
        for i in range(len(self.flux_density_lambda.wavelengths)):
            cross_section = self.absorb_cross_section_spectrum.values[i]
            int_inverse_seconds_i = (
                cross_section * self.flux_density_lambda.values[i] * H_C_COM * 1e-1
            )
            photon_rate_lambda += int_inverse_seconds_i * (
                1.0 / (1.0 + (self.fluorescent_lifetime * int_inverse_seconds_i))
            )
        print(photon_rate_lambda)

        return photon_rate_lambda  # 1/s, 10^-1 combined all conversion factors


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
        print(
            WavelengthDependentProperty(
                wavelengths=wavelengths, values=emission_rate_lambda
            )
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
        print(
            WavelengthDependentProperty(
                wavelengths=wavelengths, values=transmission_rate_lambda
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
        self.generator = [
            self.psf(self.transmission_photon_rate.wavelengths[i], self.position[2])
            for i in range(len(self.transmission_photon_rate.wavelengths))
        ]

    def incident_photons_calc(self, dt: float) -> Tuple[float, List]:
        photons = 0
        psf_hold = []
        for i in range(len(self.transmission_photon_rate.wavelengths)):
            qe_lam = self.quantumEff.get_qe(
                self.transmission_photon_rate.wavelengths[i]
            )
            photons_n = self.transmission_photon_rate.values[i] * dt
            photons += photons_n
            psf_gen = (
                self.generator[i].normalize_psf(
                    self.generator[i].gaussian_psf_z(z_val=self.position[2]), mode="sum"
                )
                * self.generator[i].gaussian_psf_z_xy0(z_val=self.position[2])
                * photons_n
            )
            psf_hold.append(photon_noise(psf_gen) * qe_lam)

        return photons, psf_hold
