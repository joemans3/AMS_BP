from typing import Dict, Iterator, List, Literal, Union

from pydantic import BaseModel


class StringNode:
    def __init__(self, value: Union[str, List["StringNode"]] = None) -> None:
        """Initialize a node with either a string or a nested list of nodes."""
        if isinstance(value, str):
            self.value = value
            self.children = None
        elif isinstance(value, list):
            self.value = None
            self.children = value
        elif value is None:
            self.value = None
            self.children = []
        else:
            raise ValueError("Value must be a string, a list of StringNode, or None.")

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (contains a string)."""
        return self.value is not None

    def add(self, child: "StringNode") -> None:
        """Add a child node to this node."""
        if self.is_leaf():
            raise TypeError("Cannot add children to a leaf node.")
        self.children.append(child)

    def remove(self, child: "StringNode") -> None:
        """Remove a child node."""
        if self.is_leaf():
            raise TypeError("Cannot remove children from a leaf node.")
        self.children.remove(child)

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        if self.is_leaf():
            return f"StringNode(value={self.value!r})"
        return f"StringNode(children={self.children!r})"

    def __iter__(self) -> Iterator["StringNode"]:
        """Iterate over children if this is not a leaf node."""
        if not self.is_leaf():
            return iter(self.children)
        raise TypeError("Leaf nodes are not iterable.")


def to_string_node(nested: Union[str, List]) -> StringNode:
    """
    Convert a nested list of strings into a StringNode structure.

    Args:
        nested: A string or a nested list of strings.

    Returns:
        A StringNode representing the nested structure.
    """
    if isinstance(nested, str):
        # Base case: If it's a string, create a leaf node
        return StringNode(nested)
    elif isinstance(nested, list):
        # Recursive case: If it's a list, create a parent node with children
        children = [to_string_node(item) for item in nested]
        return StringNode(children)
    else:
        raise ValueError("Input must be a string or a nested list of strings.")


class MetaData(BaseModel):
    notes: StringNode | list | str
    axes: str
    TimeIncrement: float
    TimeIncrementUnit: Literal["s", "ms"]
    PhysicalSizeX: float
    PhysicalSizeXUnit: Literal["nm", "m"]
    PhysicalSizeY: float
    PhysicalSizeYUnit: Literal["nm", "m"]
    Channel: Dict[Literal["Name"], List[str]]

    def __post_init__(self):
        if isinstance(self.notes, (list, str)):
            self.notes = to_string_node(self.notes)

    class Config:
        arbitrary_types_allowed = True
