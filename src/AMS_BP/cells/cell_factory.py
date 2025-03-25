import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

from boundedfbm.cells.base_cell import BaseCell

# Import cell creation functions and parameter classes
from boundedfbm.cells.ovoid_cell import OvoidCellParams, make_OvoidCell
from boundedfbm.cells.rectangular_cell import (
    RectangularCellParams,
    make_RectangularCell,
)
from boundedfbm.cells.rod_cell import RodCellParams, make_RodCell
from boundedfbm.cells.spherical_cell import SphericalCellParams, make_SphericalCell

from .budding_yeast_cell import BuddingCellParams, make_BuddingCell


# ===== Exposed Enum =====
class CellType(str, Enum):
    SPHERICAL = "SphericalCell"
    ROD = "RodCell"
    RECTANGULAR = "RectangularCell"
    OVOID = "OvoidCell"
    BUDDINGYEAST = "BuddingCell"


# ===== Internal Mappings =====
_CELL_PARAM_CLASSES: Dict[CellType, Any] = {
    CellType.SPHERICAL: SphericalCellParams,
    CellType.ROD: RodCellParams,
    CellType.RECTANGULAR: RectangularCellParams,
    CellType.OVOID: OvoidCellParams,
    CellType.BUDDINGYEAST: BuddingCellParams,
}

_CELL_CREATION_MAP: Dict[CellType, Tuple[Callable, List[str]]] = {
    CellType.SPHERICAL: (make_SphericalCell, ["center", "radius"]),
    CellType.ROD: (make_RodCell, ["center", "direction", "height", "radius"]),
    CellType.RECTANGULAR: (make_RectangularCell, ["bounds"]),
    CellType.OVOID: (
        make_OvoidCell,
        ["center", "xradius", "yradius", "zradius"],
    ),
    CellType.BUDDINGYEAST: (
        make_BuddingCell,
        [
            "center",
            "mother_radius_x",
            "mother_radius_y",
            "mother_radius_z",
            "bud_radius_x",
            "bud_radius_y",
            "bud_radius_z",
            "bud_angle",
            "bud_distance",
            "neck_radius",
        ],
    ),
}


# ===== Internal Validator =====
def _validate_parameters(
    cell_type: CellType, params: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Internal function to validate cell parameters against the expected schema."""
    param_class = _CELL_PARAM_CLASSES[cell_type]

    # Validate against expected parameters
    expected_params = {
        name for name in inspect.signature(param_class).parameters if name != "self"
    }
    unexpected = set(params.keys()) - expected_params
    missing = expected_params - set(params.keys())

    errors = []
    if unexpected:
        errors.append(f"Unexpected parameter(s): {', '.join(unexpected)}")
    if missing:
        errors.append(f"Missing parameter(s): {', '.join(missing)}")

    if errors:
        return False, errors

    try:
        param_class(**params)
        return True, []
    except ValueError as e:
        return False, str(e).split("\n")


# ===== Exposed Validation Function =====
def validate_cell_parameters(
    cell_type: Union[str, CellType], params: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate parameters for a given cell type.

    Args:
        cell_type: The cell type (string or enum)
        params: Dictionary of parameters

    Returns:
        Tuple of (is_valid, error_messages)
    """
    if isinstance(cell_type, str):
        try:
            cell_type = CellType(cell_type)
        except ValueError:
            return False, [f"Unknown cell type: {cell_type}"]

    if cell_type not in _CELL_PARAM_CLASSES:
        return False, [f"Unknown cell type: {cell_type}"]

    return _validate_parameters(cell_type, params)


# ===== Exposed Factory Function =====
def create_cell(cell_type: Union[str, CellType], params: Dict[str, Any]) -> BaseCell:
    """
    Create a validated cell instance.

    Args:
        cell_type: Cell type (string or enum)
        params: Dictionary of cell parameters.

    Returns:
        BaseCell instance.

    Raises:
        ValueError: If validation fails.
    """
    if isinstance(cell_type, str):
        try:
            cell_type = CellType(cell_type)
        except ValueError:
            raise ValueError(f"Unknown cell type: {cell_type}")

    if cell_type not in _CELL_CREATION_MAP:
        raise ValueError(f"Unknown cell type: {cell_type}")

    is_valid, errors = validate_cell_parameters(cell_type, params)
    if not is_valid:
        raise ValueError(f"Invalid {cell_type} configuration: {'; '.join(errors)}")

    func, param_keys = _CELL_CREATION_MAP[cell_type]
    return func(**{key: params[key] for key in param_keys})
