
## `util_functions.py`

This module provides utility functions for various operations such as array conversion, image saving, binning, subsegmentation, and directory structure creation.

### Functions

#### `convert_arrays_to_lists`

```python
def convert_arrays_to_lists(obj: np.ndarray | dict) -> list | dict:
    """
    Recursively convert NumPy arrays to lists.

    Parameters:
    -----------
    obj : np.ndarray | dict
        Object to be converted.

    Returns:
    --------
    list | dict
        Converted object with NumPy arrays replaced by lists.
    """
```

- **Description**: This function recursively converts NumPy arrays within a given object (either a NumPy array or a dictionary containing arrays) to lists.

#### `convert_lists_to_arrays`

```python
def convert_lists_to_arrays(obj: list | dict) -> np.ndarray | dict:
    """
    Recursively convert lists to NumPy arrays.

    Parameters:
    -----------
    obj : list | dict
        Object to be converted.

    Returns:
    --------
    np.ndarray | dict
        Converted object with lists replaced by NumPy arrays.
    """
```

- **Description**: This function recursively converts lists within a given object (either a list or a dictionary containing lists) to NumPy arrays.

#### `save_tiff`

```python
def save_tiff(
    image: np.ndarray, path: str, img_name: str | None = None, tifffile_args: dict = {}
) -> None:
    """
    Save the image as a TIFF file.

    Parameters:
    -----------
    image : np.ndarray
        Image to be saved.
    path : str
        Path where the image will be saved.
    img_name : str, optional
        Name of the image file (without extension), by default None.
    tifffile_args: dict(str, val)
        named arguments passed to the tifffile plugin as a dict.

    Returns:
    --------
    None
    """
```

- **Description**: This function saves a given image as a TIFF file to the specified path. It optionally allows for additional arguments to be passed to the `tifffile` plugin.

#### `binning_array`

```python
def binning_array(
    array_to_bin: np.ndarray, binning_size: int, mode: str = "sum"
) -> np.ndarray:
    """
    Bin an N-dimensional array by summing values in each bin.

    Parameters:
    -----------
    array_to_bin: numpy.ndarray
        Input N-dimensional array to be binned
    binning_size : int
        Size of the binning window (e.g., 2 for 2x2 binning)
    mode : str, optional
        Method for binning. Currently only supports 'sum'

    Returns:
    --------
    numpy.ndarray
        Binned array with reduced dimensions
    """
```

- **Description**: This function performs binning on an N-dimensional array by summing the values within each bin. The binning size determines the window size for binning.

#### `sub_segment`

```python
def sub_segment(
    img: np.ndarray,
    subsegment_num: int,
    img_name: str | None = None,
    subsegment_type: str = "mean",
) -> list[np.ndarray]:
    """
    Perform subsegmentation on the image.

    Parameters:
    -----------
    img : np.ndarray
        Image to be subsegmented.
    subsegment_num : int
        Number of subsegments to be created.
    img_name : str, optional
        Name of the image, by default None.
    subsegment_type : str, optional
        Type of subsegmentation to be performed. Options are "mean", "max", "std". Default is "mean".

    Returns:
    --------
    list[np.ndarray]
        List of subsegmented images.

    Raises:
    -------
    ValueError
        If the subsegment type is not supported.
    """
```

- **Description**: This function performs subsegmentation on an image by dividing it into the specified number of subsegments and applying the chosen subsegmentation type (mean, max, or standard deviation).

#### `make_directory_structure`

```python
def make_directory_structure(
    cd: str,
    img_name: str,
    img: np.ndarray,
    subsegment_type: str,
    subsegment_num: int,
    **kwargs,
) -> list[np.ndarray]:
    """
    Create the directory structure for the simulation, save the image, and perform subsegmentation.

    Parameters:
    -----------
    cd : str
        Directory where the simulation will be saved.
    img_name : str
        Name of the image.
    img : np.ndarray
        Image to be subsegmented.
    subsegment_type : str
        Type of subsegmentation to be performed.
    subsegment_num : int
        Number of subsegments to be created.
    **kwargs : dict
        Additional keyword arguments, including:
        - data : dict (optional)
            Dictionary of data to be saved, keys are "map", "tracks", "points_per_frame".
        - parameters : dict (optional)
            Parameters of the simulation to be saved.

    Returns:
    --------
    list[np.ndarray]
        List of subsegmented images.

    Raises:
    -------
    None
    """
```

- **Description**: This function creates the directory structure for a simulation, saves the image, and performs subsegmentation. It also saves additional data and parameters if provided.

#### `ms_to_seconds`

```python
@cache
def ms_to_seconds(time: int | float) -> float:
    return time * 1e-3
```

- **Description**: This function converts a time value from milliseconds to seconds. The `@cache` decorator is used to cache the result for faster access if the same input is provided again.