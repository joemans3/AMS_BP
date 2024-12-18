import json
import os

import numpy as np

from ..configio.configmodels import OutputParameters
from ..metadata.metadata import MetaData
from ..utils.util_functions import save_tiff


def save_config_frames(
    config: MetaData, frames: np.ndarray, outputparams: OutputParameters
) -> None:
    cd = outputparams.output_path
    # make the directory if it does not exist
    if not os.path.exists(cd):
        os.makedirs(cd)
    save_tiff(frames, cd, outputparams.output_name)

    # make json ster. from the MetaData
    metadata_json = config.model_dump()

    # save json
    json_path = os.path.join(cd, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata_json, f)
