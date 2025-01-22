import json
import os
import pickle
from bisect import bisect_right

import numpy as np
import skimage as skimage
from PIL import Image

from ..utils.decorators import cache


def find_le(a, x) -> int:
    "Find rightmost value less than or equal to x"
    i = bisect_right(a, x)
    if i:
        return a[i - 1]
    raise ValueError


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_arrays_to_lists(v) for k, v in obj.items()}
    else:
        return obj


# Function to recursively convert lists to NumPy arrays
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
    if isinstance(obj, list):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
    else:
        return obj


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
    if img_name is None:
        skimage.io.imsave(path, image)
    else:
        skimage.io.imsave(
            os.path.join(path, img_name + ".tiff"),
            image,
            plugin="tifffile",
            plugin_args=tifffile_args,
        )
    return


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
    if binning_size == 1:
        return array_to_bin
    if mode != "sum":
        raise ValueError("Only 'sum' mode is currently supported")

    # Get the shape of the input array
    original_shape = np.array(array_to_bin.shape)

    # Calculate the shape of the output array
    # We need to handle both cases where the dimension is divisible by binning_size
    # and where it's not
    output_shape = np.ceil(original_shape / binning_size).astype(int)

    # Calculate the effective shape that we can fully bin
    # This handles cases where the array dimensions aren't perfectly divisible
    effective_shape = (output_shape * binning_size) - original_shape

    # Pad the array if necessary to make it divisible by binning_size
    if np.any(effective_shape > 0):
        pad_width = np.array([(0, int(s)) for s in effective_shape])
        array_to_bin = np.pad(array_to_bin, pad_width, mode="constant")

    # Generate slicing for the reshape operation
    new_shape = []
    for dim in array_to_bin.shape:
        new_shape.extend([dim // binning_size, binning_size])

    # Reshape and sum along the appropriate axes
    reshaped = array_to_bin.reshape(new_shape)

    # Calculate the axes to sum over
    # For N dimensions, we want to sum over axes 1, 3, 5, etc.
    sum_axes = tuple(range(1, len(new_shape), 2))

    # Perform the binning by summing
    binned = reshaped.sum(axis=sum_axes)

    return binned


# function to perform the subsegmentation
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
    supported_subsegment_types = ["mean", "max", "std"]
    if subsegment_type not in supported_subsegment_types:
        raise ValueError(
            f"Subsegment type {subsegment_type} is not supported. Supported types are {supported_subsegment_types}"
        )
    # get the dimensions of the image
    dims = img.shape
    # get the number of frames
    num_frames = dims[0]
    # find the number of frames per subsegment
    frames_per_subsegment = int(num_frames / subsegment_num)
    hold_img = []
    for j in np.arange(subsegment_num):
        if subsegment_type == "mean":
            hold_img.append(
                np.mean(
                    img[
                        int(j * frames_per_subsegment) : int(
                            (j + 1) * frames_per_subsegment
                        )
                    ],
                    axis=0,
                )
            )
        elif subsegment_type == "max":
            hold_img.append(
                np.max(
                    img[
                        int(j * frames_per_subsegment) : int(
                            (j + 1) * frames_per_subsegment
                        )
                    ],
                    axis=0,
                )
            )
        elif subsegment_type == "std":
            hold_img.append(
                np.std(
                    img[
                        int(j * frames_per_subsegment) : int(
                            (j + 1) * frames_per_subsegment
                        )
                    ],
                    axis=0,
                )
            )
    return hold_img


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
    # make the directory if it does not exist
    if not os.path.exists(cd):
        os.makedirs(cd)
    # track_pickle
    track_pickle = os.path.join(cd, "Track_dump.pkl")
    # params_pickle
    params_pickle = os.path.join(cd, "params_dump.pkl")
    # params_json
    params_json = os.path.join(cd, "params_dump.json")

    # saves the data if it is passed as a keyword argument (map,tracks,points_per_frame)
    with open(track_pickle, "wb+") as f:
        pickle.dump(kwargs.get("data", {}), f)
    # saves the parameters used to generate the simulation
    with open(params_pickle, "wb+") as f:
        pickle.dump(kwargs.get("parameters", {}), f)

    # in this directory, dump the parameters into a json file
    with open(params_json, "w") as f:
        # dump the parameters into a json file
        # json.dump(convert_arrays_to_lists(kwargs.get("parameters", {})), f)
        json.dump({}, f)

    # make a diretory inside cd called Analysis if it does not exist
    if not os.path.exists(os.path.join(cd, "Analysis")):
        os.makedirs(os.path.join(cd, "Analysis"))
    # save the img file with its name in the cd directory
    save_tiff(img, cd, img_name=img_name)
    # make a directory inside cd called segmented if it does not exist
    if not os.path.exists(os.path.join(cd, "segmented")):
        os.makedirs(os.path.join(cd, "segmented"))
    # perform subsegmentation on the image
    hold_img = sub_segment(
        img, subsegment_num, img_name=img_name, subsegment_type=subsegment_type
    )
    # create the names for the subsegmented images
    hold_name = []
    for i in np.arange(subsegment_num):
        hold_name.append(
            os.path.join(cd, "segmented", str(int(i) + 1) + "_" + img_name + ".tif")
        )
    # save the subsegmented images
    for i in np.arange(subsegment_num):
        img = Image.fromarray(hold_img[i])
        img.save(hold_name[i])
    return hold_img


@cache
def ms_to_seconds(time: int | float) -> float:
    return time * 1e-3
