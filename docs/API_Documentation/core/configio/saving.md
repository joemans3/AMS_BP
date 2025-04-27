# Module: `saving.py`

The `saving.py` module provides functionality for saving processed data and metadata to disk. It includes methods to save image frames as TIFF files and metadata as JSON files.

## Dependencies

- `json`: For handling JSON serialization and deserialization.
- `os`: For interacting with the operating system, such as file and directory operations.
- `numpy`: For handling numerical data, particularly image frames.
- `tifffile.TiffWriter`: For writing image data to TIFF files.
- `OutputParameters`: A model from the `configio.configmodels` module, representing output parameters.
- `MetaData`: A model from the `metadata.metadata` module, representing metadata associated with the data.

## Functions

### `save_config_frames`

```python
def save_config_frames(
    config: MetaData, frames: np.ndarray, outputparams: OutputParameters
) -> None:
```

Saves the provided image frames and associated metadata to disk.

#### Parameters

- **config** (`MetaData`): 
  - Metadata associated with the image frames. This includes information such as axes definitions and other relevant metadata.
  
- **frames** (`np.ndarray`): 
  - A NumPy array containing the image frames to be saved.
  
- **outputparams** (`OutputParameters`): 
  - An instance of `OutputParameters` containing the output path and file name.

#### Returns

- **None**

#### Behavior

1. **Directory Creation**: 
   - If the directory specified in `outputparams.output_path` does not exist, it will be created.

2. **TIFF File Writing**: 
   - The image frames are saved as a TIFF file using the `TiffWriter` class from the `tifffile` library. The file is saved with the name specified in `outputparams.output_name` and the `.ome.tiff` extension. The `bigtiff=True` parameter ensures that the file can handle large datasets.
   - The metadata from the `config` object, excluding any notes, is embedded in the TIFF file.

3. **JSON Metadata Saving**: 
   - The complete metadata from the `config` object is serialized to a JSON file named `metadata.json` in the same directory as the TIFF file.

#### Example

```python
from ..configio.configmodels import OutputParameters
from ..metadata.metadata import MetaData
import numpy as np

# Example usage
config = MetaData(axes="TYX", notes="Example metadata")
frames = np.random.rand(10, 256, 256)  # Example frames
outputparams = OutputParameters(output_path="./output", output_name="example")

save_config_frames(config, frames, outputparams)
```

This will save the frames as `example.ome.tiff` in the `./output` directory and the metadata as `metadata.json` in the same directory.

## Notes

- The function assumes that the `MetaData` and `OutputParameters` models have appropriate methods (`model_dump`) for serializing their data.
- The `TiffWriter` class is used for writing TIFF files, and it is assumed that the `tifffile` library is installed and properly configured.
- The function handles large datasets by using the `bigtiff=True` parameter, which is necessary for datasets that exceed the 4GB limit of standard TIFF files.