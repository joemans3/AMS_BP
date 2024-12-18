from typing import List, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator


class CellParameters(BaseModel):
    cell_space: List[List[float]] = Field(description="Cell space dimensions in um")
    cell_axial_radius: float = Field(description="Axial radius in um")

    @field_validator("cell_space")
    def convert_cell_space(cls, v):
        return np.array(v)


class MoleculeParameters(BaseModel):
    num_molecules: int
    track_type: Literal["fbm", "constant"] = Field(
        description="Type of molecular motion, either fbm or constant"
    )
    diffusion_coefficient: List[float] = Field(
        description="Diffusion coefficients in um^2/s"
    )
    diffusion_track_amount: List[float]
    hurst_exponent: List[float]
    hurst_track_amount: List[float]
    allow_transition_probability: bool
    transition_matrix_time_step: int = Field(description="Time step in ms")
    diffusion_transition_matrix: List[List[float]]
    hurst_transition_matrix: List[List[float]]
    state_probability_diffusion: List[float]
    state_probability_hurst: List[float]

    @field_validator(
        "diffusion_coefficient",
        "diffusion_track_amount",
        "hurst_exponent",
        "hurst_track_amount",
        "state_probability_diffusion",
        "state_probability_hurst",
    )
    def convert_to_array(cls, v):
        return np.array(v)

    @field_validator("diffusion_transition_matrix", "hurst_transition_matrix")
    def convert_matrix_to_array(cls, v):
        return np.array(v)


class GlobalParameters(BaseModel):
    sample_plane_dim: List[float] = Field(description="Sample plane dimensions in um")
    cycle_count: int
    exposure_time: int = Field(description="Exposure time in ms")
    interval_time: int = Field(description="Interval time in ms")
    oversample_motion_time: int = Field(description="Oversample motion time in ms")

    @field_validator("sample_plane_dim")
    def convert_sample_plane_dim(cls, v):
        return np.array(v)


class CondensateParameters(BaseModel):
    initial_centers: List[List[float]] = Field(description="Initial centers in um")
    initial_scale: List[float] = Field(description="Initial scale in um")
    diffusion_coefficient: List[float] = Field(
        description="Diffusion coefficients in um^2/s"
    )
    hurst_exponent: List[float]
    density_dif: int

    @field_validator(
        "initial_centers", "initial_scale", "diffusion_coefficient", "hurst_exponent"
    )
    def convert_to_array(cls, v):
        return np.array(v)


class OutputParameters(BaseModel):
    output_path: str
    output_name: str
    subsegment_type: str
    subsegment_number: int


class ConfigList(BaseModel):
    CellParameters: CellParameters
    MoleculeParameters: MoleculeParameters
    GlobalParameters: GlobalParameters
    CondensateParameters: CondensateParameters
    OutputParameters: OutputParameters
