from typing import List

from pydantic import BaseModel

from ....optics.filters.filters import FilterSet


class Channels(BaseModel):
    filtersets: List[FilterSet]
    num_channels: int
    splitting_efficiency: List[float]
    names: List[str]

    # @field_validator("num_channels")
    # def validate_num_channels(cls, v, info):
    #     print(info.data.keys())
    #     filtersets = info.data.get("filtersets", [])
    #     splitting_efficiency = info.data.get("splitting_efficiency", [])
    #     if v != len(filtersets):
    #         raise ValueError(
    #             f"num_channels ({v}) must equal the length of filtersets ({len(filtersets)})"
    #         )
    #     if v != len(splitting_efficiency):
    #         raise ValueError(
    #             f"num_channels ({v}) must equal the length of splitting_efficiency ({len(splitting_efficiency)})"
    #         )
    #     return v
