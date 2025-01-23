import math
from functools import partial
from typing import Callable, List, Tuple

import numpy as np

"""Currently unused module"""


def plane_point_scan(
    x_lims: List[float],
    y_lims: List[float],
    step_xy: float,
) -> np.ndarray:
    """
    Generate a point scanning pattern for a confocal microscope plane scan.

    Args:
        x_lims (List[float]): [min_x, max_x] scanning limits in x direction
        y_lims (List[float]): [min_y, max_y] scanning limits in y direction
        step_xy (float): Step size between points in both x and y directions

    Returns:
        np.ndarray: Array of shape (n_points, 2) containing [x, y] coordinates for the scan
    """
    # Calculate number of points in each dimension
    nx = math.ceil((x_lims[1] - x_lims[0]) / step_xy) + 1
    ny = math.ceil((y_lims[1] - y_lims[0]) / step_xy) + 1

    # Generate coordinate arrays
    x = np.linspace(x_lims[0], x_lims[1], nx)
    y = np.linspace(y_lims[0], y_lims[1], ny)

    # Create meshgrid for all coordinates
    xx, yy = np.meshgrid(x, y)

    # Convert to scan pattern array
    # For even rows, reverse x direction for serpentine scan
    scan_points = []
    for i in range(ny):
        row_x = xx[i]
        row_y = yy[i]

        if i % 2 == 1:  # Reverse even rows for serpentine pattern
            row_x = row_x[::-1]

        points = np.column_stack((row_x, row_y))
        scan_points.append(points)

    # Combine all points into final array
    scan_pattern = np.vstack(scan_points)

    return scan_pattern


def confocal_pointscan_time_z(
    x_lims: List[float],
    y_lims: List[float],
    step_xy: float,  # can be defined as the beam width at the focus plane
    frame_exposure_time: float,  # s
) -> Tuple[Callable[[float, float], Tuple[float, float, float]], float]:
    scan_pattern = plane_point_scan(x_lims=x_lims, y_lims=y_lims, step_xy=step_xy)
    scan_pattern_len = len(scan_pattern)

    dwell_time = frame_exposure_time / scan_pattern_len

    def return_laser_position(
        z_position: float, time: float
    ) -> Tuple[float, float, float]:
        index_frame = time % frame_exposure_time
        ind = int(index_frame / dwell_time)
        # print(index_frame, ind)
        return (*scan_pattern[ind], z_position)

    return return_laser_position, dwell_time


def confocal_pointscan_time_z0(
    x_lims: List[float],
    y_lims: List[float],
    step_xy: float,  # can be defined as the beam width at the focus plane
    frame_exposure_time: float,  # s
    z_val: float,  # um
) -> Tuple[Callable[[float], Tuple[float, float, float]], float]:
    """
    Create a generator for a point scanning pattern for a confocal microscope plane scan which takes in a time and returns the postion of the laser.

    Args:
        x_lims (List[float]): [min_x, max_x] scanning limits in x direction
        y_lims (List[float]): [min_y, max_y] scanning limits in y direction
        step_xy (float): Step size between points in both x and y directions
        frame_exposure_time (float): exposure time of the frame
        z_val (float): z value of the sample plane

    Returns:
        Callable[time]: (x,y,z) position of the laser
        dwell_time (float): the dwell time per position
    """
    func, dwell_time = confocal_pointscan_time_z(
        x_lims, y_lims, step_xy, frame_exposure_time
    )
    return partial(func, z_val), dwell_time
