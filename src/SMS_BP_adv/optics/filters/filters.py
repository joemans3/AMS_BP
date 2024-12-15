from functools import cache
from typing import Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

CustomNDarray: TypeAlias = NDArray[np.float64]


class FilterSpectrum(BaseModel):
    """Represents the spectral characteristics of an optical filter"""

    wavelengths: NDArray[np.float64] = Field(description="Wavelengths in nanometers")
    transmission: NDArray[np.float64] = Field(description="Transmission values (0-1)")
    name: str

    @field_validator("transmission")
    def validate_transmission(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if not np.all((v >= 0) & (v <= 1)):
            raise ValueError("Transmission values must be between 0 and 1")
        return v

    @field_validator("wavelengths")
    def validate_wavelengths(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if not np.all(v > 0):
            raise ValueError("Wavelengths must be positive")
        return v

    @field_validator("wavelengths", "transmission")
    def validate_array_lengths(
        cls, v: NDArray[np.float64], info
    ) -> NDArray[np.float64]:
        if (
            info.data.get("wavelengths") is not None
            and info.data.get("transmission") is not None
        ):
            if len(info.data["wavelengths"]) != len(info.data["transmission"]):
                raise ValueError(
                    "Wavelengths and transmission arrays must have the same length"
                )
        return v

    class Config:
        arbitrary_types_allowed = True

    def find_transmission(self, wavelength: float) -> float:
        return np.interp(wavelength, self.wavelengths, self.transmission)


class FilterSet(BaseModel):
    """Represents a complete filter set (excitation, dichroic, emission)"""

    excitation: FilterSpectrum
    dichroic: FilterSpectrum
    emission: FilterSpectrum
    name: str = Field(default="Generic Filter Set")

    @cache
    def get_total_transmission(self, wavelength: float) -> float:
        """Calculate total transmission at a specific wavelength"""
        # Interpolate transmission values for each filter component
        exc_trans = np.interp(
            wavelength, self.excitation.wavelengths, self.excitation.transmission
        )
        dic_trans = np.interp(
            wavelength, self.dichroic.wavelengths, self.dichroic.transmission
        )
        em_trans = np.interp(
            wavelength, self.emission.wavelengths, self.emission.transmission
        )

        return exc_trans * dic_trans * em_trans


def create_bandpass_filter(
    center_wavelength: float,
    bandwidth: float,
    transmission_peak: float = 0.95,
    points: int = 1000,
    name: Optional[str] = None,
) -> FilterSpectrum:
    """
    Create a gaussian-shaped bandpass filter

    Args:
        center_wavelength: Center wavelength in nm
        bandwidth: FWHM bandwidth in nm
        transmission_peak: Peak transmission (0-1)
        points: Number of points in the spectrum
        name: Optional name for the filter
    """
    if name is None:
        name = f"BP{center_wavelength}/{bandwidth}"

    wavelengths = np.linspace(
        center_wavelength - 2 * bandwidth, center_wavelength + 2 * bandwidth, points
    )

    sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))
    transmission = transmission_peak * np.exp(
        -((wavelengths - center_wavelength) ** 2) / (2 * sigma**2)
    )

    return FilterSpectrum(wavelengths=wavelengths, transmission=transmission, name=name)


def create_tophat_filter(
    center_wavelength: float,
    bandwidth: float,
    transmission_peak: float = 0.95,
    edge_steepness: float = 5.0,
    points: int = 1000,
    name: Optional[str] = None,
) -> FilterSpectrum:
    """
    Create a top-hat (rectangular) shaped filter with smooth edges

    Args:
        center_wavelength: Center wavelength in nm
        bandwidth: Width of the passband in nm (FWHM)
        transmission_peak: Peak transmission (0-1)
        edge_steepness: Controls the sharpness of the edges (higher = sharper)
        points: Number of points in the spectrum
        name: Optional name for the filter
    """
    if name is None:
        name = f"TH{center_wavelength}/{bandwidth}"

    # Create wavelength array with some padding
    wavelengths = np.linspace(
        center_wavelength - 2 * bandwidth, center_wavelength + 2 * bandwidth, points
    )

    # Create smooth edges using sigmoid function
    left_edge = 1 / (
        1
        + np.exp(-edge_steepness * (wavelengths - (center_wavelength - bandwidth / 2)))
    )
    right_edge = 1 / (
        1 + np.exp(edge_steepness * (wavelengths - (center_wavelength + bandwidth / 2)))
    )

    # Combine edges to form top-hat
    transmission = transmission_peak * left_edge * right_edge

    return FilterSpectrum(wavelengths=wavelengths, transmission=transmission, name=name)


def create_allow_all_filter(points: int, name: Optional[str] = None) -> FilterSpectrum:
    """
    Create a filter that allows all wavelengths

    Args:
        name: Optional name for the filter
    """
    if name is None:
        name = "Allow All"

    wavelengths = np.linspace(300, 800, points)
    transmission = np.ones_like(wavelengths)

    return FilterSpectrum(wavelengths=wavelengths, transmission=transmission, name=name)
