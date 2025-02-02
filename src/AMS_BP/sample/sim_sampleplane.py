from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..sample.flurophores.flurophore_schema import WavelengthDependentProperty
from .flurophores import Fluorophore, State, StateTransition

# can reuse. Used for indicating that there is no transmittance at a given wavelenght ( any given time in which the flurophore is NOT in the fluorescent state). Also used for t=0.
EMPTY_STATE_HISTORY_DICT = {
    "Empty": WavelengthDependentProperty(wavelengths=[], values=[])
}


@dataclass
class FluorescentObject:
    """Represents a fluorescent object in the sample plane"""

    object_id: str
    position: Tuple[float, float, float]  # (x, y, z) in μm
    fluorophore: Fluorophore  # Fluorophore parameters

    # Track position history: Dict[timestamp_ms, position]
    position_history: Optional[Dict[int, Tuple[float, float, float]]] = None
    # Track fluorophore state history: Dict[timestamp_ms, (current_state, random_val, available_transitions)] # never input, calculated post.
    state_history: Optional[
        Dict[
            int,
            Tuple[State, Dict[str, WavelengthDependentProperty], List[StateTransition]],
        ]
    ] = None

    def __post_init__(self):
        if self.position_history is None:
            self.position_history = {0: self.position}  # Initialize at t=0

        if self.state_history is None:
            # use the initial state of the fluorophore opbject
            initial_state = self.fluorophore.initial_state
            initial_state_transitions = self.fluorophore._find_transitions(
                initial_state.name
            )

            # select dark state [Optional]
            # initial_state = next(
            #     state
            #     for state in self.fluorophore.states.values()
            #     if state.state_type == StateType.DARK
            # )
            # initial_transitions = [
            #     t
            #     for t in self.fluorophore.transitions.values()
            #     if t.from_state == initial_state.name
            # ]

            self.state_history = {
                0: (
                    initial_state,
                    EMPTY_STATE_HISTORY_DICT,  # placeholder
                    initial_state_transitions,
                )
            }


@dataclass
class CellSpaceView:
    """Represents a 3D field of view with bounds in x, y, and z dimensions"""

    x_bounds: Tuple[float, float]
    y_bounds: Tuple[float, float]
    z_bounds: Tuple[float, float]

    def __post_init__(self):
        """Validate bounds after initialization"""
        for name, bounds in [
            ("x", self.x_bounds),
            ("y", self.y_bounds),
            ("z", self.z_bounds),
        ]:
            if bounds[0] >= bounds[1]:
                raise ValueError(f"{name} bounds must have min < max")

    def contains(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is within the field of view

        Args:
            position: (x, y, z) position in μm

        Returns:
            bool: True if position is within FOV bounds
        """
        x, y, z = position
        return (
            self.x_bounds[0] <= x <= self.x_bounds[1]
            and self.y_bounds[0] <= y <= self.y_bounds[1]
            and self.z_bounds[0] <= z <= self.z_bounds[1]
        )

    @property
    def bounds(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return the FOV bounds"""
        return (self.x_bounds, self.y_bounds, self.z_bounds)


@dataclass
class SampleSpace:
    """
    Represents the total 3D space available for simulation.
    All coordinates are in μm and the space is defined from (0,0,-z_min) to (x_max,y_max,z_max)
    """

    x_max: float
    y_max: float
    z_min: float
    z_max: float

    def contains_fov(self, fov: CellSpaceView) -> bool:
        """
        Check if a field of view fits within the sample space

        Args:
            fov: FieldOfView instance to check

        Returns:
            bool: True if FOV is completely within sample space
        """
        x_min, x_max = fov.x_bounds
        y_min, y_max = fov.y_bounds
        z_min, z_max = fov.z_bounds

        return (
            x_min <= x_min <= x_max <= self.x_max
            and x_min <= y_min <= y_max <= self.y_max
            and self.z_min <= z_min <= z_max <= self.z_max
        )

    def contains_position(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is within the sample space

        Args:
            position: (x, y, z) position in μm

        Returns:
            bool: True if position is within sample space
        """
        x, y, z = position
        return (
            0 <= x <= self.x_max
            and 0 <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )

    @property
    def bounds(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return the space bounds"""
        return ((0, self.x_max), (0, self.y_max), (self.z_min, self.z_max))


class SamplePlane:
    def __init__(
        self,
        sample_space: SampleSpace,
        fov: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        oversample_motion_time: int,  # ms
        t_end: int,  # ms
    ):
        """
        Initialize the sample plane

        Args:
            sample_space: SampleSpace instance defining the total simulation volume
            fov: Field of view bounds as ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in μm
            oversample_motion_time: Time step in milliseconds
            t_end: End time in milliseconds
        """
        self._space = sample_space
        self._fov = CellSpaceView(fov[0], fov[1], fov[2])

        # Validate that FOV is within sample space
        if not self._space.contains_fov(self._fov):
            raise ValueError("Field of view must be within sample space bounds")

        self._objects: Dict[str, FluorescentObject] = {}
        self._spatial_index: Dict[int, Dict[Tuple[int, int, int], List[str]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self._grid_size: float = 1.0  # μm per grid cell

        # Time parameters
        self.dt = oversample_motion_time  # ms
        self.dt_s = self.dt * (1e-3)
        self.t_end = t_end  # ms
        self.time_points = list(range(0, self.t_end, self.dt))

    def add_object(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        fluorophore: Fluorophore,
        trajectory: Optional[Dict[int, Tuple[float, float, float]]] = None,
    ) -> bool:
        """Add a fluorescent object to the sample plane"""
        # # Verify position is within sample space
        # if not self._space.contains_position(position):
        #     return False
        #
        # # Verify initial position is within FOV
        # if not self.is_within_fov(position):
        #     return False

        # # If trajectory provided, verify all positions
        # if trajectory is not None:
        #     if not all(
        #         self._space.contains_position(pos) for pos in trajectory.values()
        #     ):
        #         return False
        #     if not all(self.is_within_fov(pos) for pos in trajectory.values()):
        #         return False
        #
        if object_id in self._objects:
            return False

        # Initialize position history
        if trajectory is None:
            # If no trajectory provided, object stays at initial position
            trajectory = {t: position for t in self.time_points}
        else:
            # Verify trajectory timestamps align with time points
            if not all(t in self.time_points for t in trajectory.keys()):
                raise ValueError("Trajectory timestamps must align with time points")

        obj = FluorescentObject(object_id, position, fluorophore, trajectory)
        self._objects[object_id] = obj

        # # Update spatial index for all time points
        # for t in self.time_points:
        #     pos = trajectory[t]
        #     self._update_spatial_index(object_id, pos, t)
        #
        return True

    def get_all_objects(self) -> List[FluorescentObject]:
        """Get all objects in the sample plane"""
        return list(self._objects.values())

    def get_objects_in_range(
        self, center: Tuple[float, float, float], radius: float, t: int
    ) -> List[FluorescentObject]:
        """
        Get all objects within a specified radius of a point at a given time

        Args:
            center: Center point (x, y, z) in μm
            radius: Search radius in μm
            t: Timestamp in ms
        """
        if t not in self.time_points:
            raise ValueError(f"Invalid time point: {t}")

        grid_center = self._get_grid_position(center)
        search_radius = int(radius / self._grid_size) + 1

        nearby_objects = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    grid_pos = (
                        grid_center[0] + dx,
                        grid_center[1] + dy,
                        grid_center[2] + dz,
                    )
                    for obj_id in self._spatial_index[t][grid_pos]:
                        obj = self._objects[obj_id]
                        if self._distance(center, obj.position_history[t]) <= radius:
                            nearby_objects.append(obj)
        return nearby_objects

    def is_within_fov(self, position: Tuple[float, float, float]) -> bool:
        """Check if a position is within the FOV"""
        return self._fov.contains(position)

    def _update_spatial_index(
        self, object_id: str, position: Tuple[float, float, float], t: int
    ):
        """Update the spatial index for an object at a specific time"""
        grid_pos = self._get_grid_position(position)
        self._spatial_index[t][grid_pos].append(object_id)

    def get_object(self, object_id: str) -> Optional[FluorescentObject]:
        """Retrieve an object by its ID"""
        return self._objects.get(object_id)

    def __iter__(self) -> Iterator[FluorescentObject]:
        """Make the sample plane iterable"""
        return iter(self._objects.values())

    def __len__(self) -> int:
        """Return the number of objects in the sample plane"""
        return len(self._objects)

    # Helper methods
    def _get_grid_position(
        self, position: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """Convert a position to grid coordinates"""
        return tuple(int(p / self._grid_size) for p in position)

    @staticmethod
    def _distance(
        p1: Tuple[float, float, float], p2: Tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    @property
    def fov_bounds(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return the FOV bounds"""
        return self._fov.bounds

    @property
    def sample_space_bounds(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return the sample space bounds"""
        return self._space.bounds


def _FOV_lims(
    xyoffset: Tuple[float, float],
    detector_pixelcount: Tuple[int, int],
    detector_pixelsize_magnification: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Utility to determine the FOV (what regions in the sample space the camera can capture) -> mainly used for defining laser position callables.

    Parameters:
    -----------
    xyoffset: Tuple[float, float]
        position in the sample plane which defines the bottom left corner on the detector pixel array; in units of the sample space (um)

    detector_pixelcount: Tuple[int, int]
        number of pixels in the detector in the both x and y

    detector_pixelsize_magnification: float
        the pixel size of each pixel (of detector_pixelcount) after the magnification of the optical setup is considered. I.e what each pixel in the final image represents in the units of the sample space (um).

    Returns:
    --------
    (x_lims, y_lims): Tuple[Tuple[float, float], Tuple[float, float]]
        min < max for each lim
    """
    x_min = xyoffset[0]
    y_min = xyoffset[1]
    x_max = x_min + detector_pixelcount[0] * detector_pixelsize_magnification
    y_max = y_min + detector_pixelcount[1] * detector_pixelsize_magnification
    return ((x_min, x_max), (y_min, y_max))
