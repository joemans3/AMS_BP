import json
import os

import numpy as np
from tifffile import TiffWriter

from ..configio.configmodels import OutputParameters
from ..metadata.metadata import MetaData


def save_config_frames(
    config: MetaData, frames: np.ndarray, outputparams: OutputParameters
) -> None:
    cd = outputparams.output_path
    # make the directory if it does not exist
    if not os.path.exists(cd):
        os.makedirs(cd)

    with TiffWriter(
        os.path.join(cd, outputparams.output_name + ".ome" + ".tiff"), bigtiff=True
    ) as f:
        f.write(
            frames,
            metadata=config.model_dump(exclude={"notes"}),
        )
    # make json ster. from the MetaData
    metadata_json = config.model_dump()

    # save json
    json_path = os.path.join(cd, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata_json, f)
