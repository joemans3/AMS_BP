from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..cells import CellType, validate_cell_parameters


class CellParameters(BaseModel):
    cell_type: Union[str, CellType]
    params: Dict[str, Any]

    def model_post_init(self, __context):
        is_valid = validate_cell_parameters(self.cell_type, self.params)
        if not is_valid:
            raise ValueError(f"Cell model creation unsuccessful: {is_valid[1]}")


class MoleculeParameters(BaseModel):
    num_molecules: List[int]
    track_type: List[Literal["fbm", "constant"]] = Field(
        description="Type of molecular motion, either fbm or constant"
    )
    diffusion_coefficient: List[List[float]] = Field(
        description="Diffusion coefficients in um^2/s"
    )
    hurst_exponent: List[List[float]]
    allow_transition_probability: List[bool]
    transition_matrix_time_step: List[int] = Field(description="Time step in ms")
    diffusion_transition_matrix: List[List[List[float]]]
    hurst_transition_matrix: List[List[List[float]]]
    state_probability_diffusion: List[List[float]]
    state_probability_hurst: List[List[float]]

    # @field_validator(
    #     "diffusion_coefficient",
    #     "hurst_exponent",
    #     "state_probability_diffusion",
    #     "state_probability_hurst",
    # )
    # def convert_to_array(cls, v):
    #     return np.array(v)
    #
    # @field_validator("diffusion_transition_matrix", "hurst_transition_matrix")
    # def convert_matrix_to_array(cls, v):
    #     return np.array(v)


class GlobalParameters(BaseModel):
    sample_plane_dim: List[float] = Field(description="Sample plane dimensions in um")
    cycle_count: int
    exposure_time: int = Field(description="Exposure time in ms")
    interval_time: int = Field(description="Interval time in ms")
    oversample_motion_time: int = Field(description="Oversample motion time in ms")

    @field_validator("sample_plane_dim")
    def convert_sample_plane_dim(cls, v):
        return np.array(v)

    def model_post_init(self, __context):
        if self.oversample_motion_time <= 0:
            raise ValueError("Oversample Motion Time must be larger than 0 ms")
        if self.exposure_time < self.oversample_motion_time:
            if self.exposure_time > 0:
                raise ValueError(
                    "Exposure time must be equal to or larger than Oversample Motion Time, or 0"
                )
        if self.interval_time < self.oversample_motion_time:
            if self.interval_time > 0:
                raise ValueError(
                    "Interval time must be equal to or larger than Oversample Motion Time, or 0"
                )


class CondensateParameters(BaseModel):
    initial_centers: List[List[List[float]]] = Field(
        description="Initial centers in um"
    )
    initial_scale: List[List[float]] = Field(description="Initial scale in um")
    diffusion_coefficient: List[List[float]] = Field(
        description="Diffusion coefficients in um^2/s"
    )
    hurst_exponent: List[List[float]]
    density_dif: List[int]

    # @field_validator(
    #     "initial_centers", "initial_scale", "diffusion_coefficient", "hurst_exponent"
    # )
    # def convert_to_array(cls, v):
    #     return np.array(v)
    #


class OutputParameters(BaseModel):
    output_path: str
    output_name: str
    subsegment_type: str
    subsegment_number: int


class ConfigList(BaseModel):
    CellParameter: Optional[CellParameters] = None
    MoleculeParameter: Optional[MoleculeParameters] = None
    GlobalParameter: GlobalParameters
    CondensateParameter: Optional[CondensateParameters] = None
    OutputParameter: Optional[OutputParameters] = None
