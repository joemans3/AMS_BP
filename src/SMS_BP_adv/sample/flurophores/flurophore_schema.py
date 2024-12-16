from enum import Enum
from typing import Callable, List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ...utils.constants import H_C_COM, N_A

NumericType = TypeVar("NumericType", float, np.ndarray, List[float])

CS_COEFF = np.log(10) * (10**3) / N_A


def normalize_values(values: List[float]) -> List[float]:
    """Normalize values to sum to 1"""
    total = sum(values)
    return [val / total for val in values]


class WavelengthDependentBase(BaseModel):
    """Base class for wavelength-dependent data"""

    wavelengths: List[float] = Field(..., description="Wavelengths (nm)")
    values: List[float] = Field(..., description="Values")

    @field_validator("wavelengths", "values")
    @classmethod
    def validate_lengths(cls, v, info):
        if "wavelengths" in info.data and len(info.data["wavelengths"]) != len(v):
            raise ValueError(f"Wavelengths and {info.field_name} must have same length")
        return v

    def get_value(self, wavelength: float) -> float:
        """Get interpolated value at a specific wavelength"""
        return np.interp(wavelength, self.wavelengths, self.values)  # pyright: ignore


class SpectralData(WavelengthDependentBase):
    """Wavelength-dependent spectral data"""

    values: List[float] = Field(
        ..., description="Intensities (0-1)", alias="intensities"
    )

    def get_intensity(self, wavelength: float) -> float:
        """Get intensity at a specific wavelength"""
        return self.get_value(wavelength)


class WavelengthDependentProperty(WavelengthDependentBase):
    """Wavelength-dependent property"""

    pass


class StateType(Enum):
    """Types of fluorophore states"""

    FLUORESCENT = "fluorescent"
    DARK = "dark"
    TRIPLET = "triplet"
    BLEACHED = "bleached"


class State(BaseModel):
    """Single state of a fluorophore"""

    name: str
    state_type: StateType
    # Spectral properties (only required for fluorescent states)
    excitation_spectrum: Optional[SpectralData] = None
    emission_spectrum: Optional[SpectralData] = None

    quantum_yield_lambda_val: Optional[float | int] = Field(
        None, ge=0, le=1, description="Quantum yield at wavelength (0-1)"
    )  # value of quantum_yield at wavelength (0-1) # at em_max
    quantum_yield: Optional[WavelengthDependentProperty] = Field(
        None, ge=0, le=1, description="Quantum yield (0-1)"
    )  # post init value of quantum_yield at various wavelengths

    extinction_coefficient_lambda_val: Optional[float] = Field(
        None, gt=0, description="M⁻¹cm⁻¹"
    )  # value of extinction_coefficient at wavelength (M⁻¹cm⁻¹) # at ex_max
    extinction_coefficient: Optional[WavelengthDependentProperty] = Field(
        None, gt=0, description="M⁻¹cm⁻¹"
    )  # post init value of extinction_coefficient at various wavelengths

    ex_max: Optional[float] = None  # wavelength in nm at max ex val
    em_max: Optional[float] = None  # wavelength in nm at max em val
    molar_cross_section: Optional[WavelengthDependentProperty] = Field(
        None, gt=0, description="cm²"
    )  # post init value of molar_cross_section at various wavelengths

    def model_post_init(self, __context):
        # populate ex_max and em_max:
        if self.excitation_spectrum is not None:
            self.ex_max = self.excitation_spectrum.wavelengths[
                self.excitation_spectrum.values.index(
                    max(self.excitation_spectrum.values)
                )
            ]

            if self.extinction_coefficient_lambda_val is not None:
                self.extinction_coefficient = self._val_ratio_expand(
                    self.extinction_coefficient_lambda_val,
                    self.ex_max,
                    self.excitation_spectrum,
                )

                _ = WavelengthDependentProperty(
                    wavelengths=self.extinction_coefficient.wavelengths,
                    values=[i * CS_COEFF for i in self.extinction_coefficient.values],
                )
                self.molar_cross_section = _

        if self.emission_spectrum is not None:
            self.em_max = self.emission_spectrum.wavelengths[
                self.emission_spectrum.values.index(max(self.emission_spectrum.values))
            ]

            if self.quantum_yield_lambda_val is not None:
                self.quantum_yield = self._val_ratio_expand(
                    self.quantum_yield_lambda_val, self.em_max, self.emission_spectrum
                )

    def _val_ratio_expand(
        self,
        val: float,
        wavelength: float,
        base_spectrum: WavelengthDependentBase,
    ) -> WavelengthDependentProperty:
        """Expand a value to a spectrum"""
        ratio = val / base_spectrum.get_value(wavelength)
        val_out = [
            ratio * base_spectrum.get_value(wl) for wl in base_spectrum.wavelengths
        ]
        wavelengths = base_spectrum.wavelengths

        return WavelengthDependentProperty(wavelengths=wavelengths, values=val_out)


class StateTransition(BaseModel):
    """Transition between states"""

    from_state: str
    to_state: str
    activation_spectrum: Optional[SpectralData] = Field(
        None, description="Wavelength-dependent activation spectrum"
    )
    activation_extinction_coefficient_lambda_val: Optional[float] = Field(
        None,
        description="Value of activation_extinction_coefficient at wavelength (M⁻¹cm⁻¹)",
    )
    activation_extinction_coefficient: Optional[WavelengthDependentProperty] = Field(
        None,
        description="Wavelength-dependent activation extinction coefficient (M⁻¹cm⁻¹)",
    )
    activation_cross_section: Optional[WavelengthDependentProperty] = Field(
        None, description="Wavelength-dependent activation cross section (cm²)"
    )  # post init value of activation_cross_section at various wavelengths
    base_rate: Optional[float] = Field(None, description="Base transition rate (1/s)")

    def model_post_init(self, __context) -> None:
        if (
            self.activation_extinction_coefficient_lambda_val is not None
            and self.activation_spectrum is not None
        ):
            # find the max wavelength of the activation spectrum
            activation_max = self.activation_spectrum.wavelengths[
                self.activation_spectrum.values.index(
                    max(self.activation_spectrum.values)
                )
            ]
            act_ext_c_spec = self._val_ratio_expand(
                self.activation_extinction_coefficient_lambda_val,
                activation_max,
                self.activation_spectrum,
            )
            self.activation_extinction_coefficient = act_ext_c_spec

            _ = WavelengthDependentProperty(
                wavelengths=self.activation_extinction_coefficient.wavelengths,
                values=[
                    i * CS_COEFF for i in self.activation_extinction_coefficient.values
                ],
            )
            self.activation_cross_section = _

    def activation_rate(self) -> Callable:
        """Get activation rate at a specific wavelength (nm) and corresponding intensity (W/um^2)"""
        if (
            self.activation_extinction_coefficient is None
            and self.activation_spectrum is None
        ):

            def _activation_rate_base(wavelength: float, intensity: float) -> float:
                if self.base_rate is None:
                    return 0
                return self.base_rate

            return _activation_rate_base  # 1/s
        else:

            @np.vectorize
            def _activation_rate(
                wavelength: float, intensity: float
            ) -> float:  # wavelength in nm, intensity in W/um^2. Final result in 1/s
                return (
                    self.activation_cross_section.get_value(wavelength)
                    * intensity
                    * self.activation_spectrum.get_intensity(wavelength)  # pyright: ignore
                    * wavelength
                    * H_C_COM  # (cross section in cm², intensity in W/um², wavelength in nm: 10^-4, 10^12, 10^-9 -> to m.)
                )

            return _activation_rate  # 1/s

    def _val_ratio_expand(
        self,
        val: float,
        wavelength: float,
        base_spectrum: WavelengthDependentBase,
    ) -> WavelengthDependentProperty:
        """Expand a value to a spectrum"""
        ratio = val / base_spectrum.get_value(wavelength)
        val_out = [
            ratio * base_spectrum.get_value(wl) for wl in base_spectrum.wavelengths
        ]
        wavelengths = base_spectrum.wavelengths

        return WavelengthDependentProperty(wavelengths=wavelengths, values=val_out)


class Fluorophore(BaseModel):
    """Complete fluorophore model"""

    name: str
    states: dict[str, State] = Field(...)  # str = State.name
    transitions: dict[
        str, StateTransition
    ]  # str = StateTransition.from_state + StateTransiiton.to_state

    @field_validator("states")
    @classmethod
    def validate_states(cls, v):
        # Must have at least one fluorescent state
        if not any(state.state_type == StateType.FLUORESCENT for state in v.values()):
            raise ValueError("At least one fluorescent state required")

        # Check for unique state names
        state_names = [state.name for state in v.values()]
        if len(state_names) != len(set(state_names)):
            raise ValueError("State names must be unique")

        return v

    @field_validator("transitions")
    @classmethod
    def validate_transitions(cls, v, info):
        if "states" not in info.data:
            return v

        state_names = {state.name for state in info.data["states"].values()}

        for transition in v.values():
            if transition.from_state not in state_names:
                raise ValueError(f"Invalid from_state: {transition.from_state}")
            if transition.to_state not in state_names:
                raise ValueError(f"Invalid to_state: {transition.to_state}")

        return v