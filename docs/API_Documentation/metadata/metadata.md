# Module: `metadata.py`

This module defines a `StringNode` class for representing hierarchical string structures and a `MetaData` class for storing metadata information. The `StringNode` class allows for the creation of tree-like structures where each node can either contain a string value or a list of child nodes. The `MetaData` class uses `pydantic` for data validation and includes fields for various metadata attributes.

## Classes

### `StringNode`

A class representing a node in a hierarchical structure of strings. Each node can either contain a string value or a list of child nodes.

#### Methods

- **`__init__(self, value: Union[str, List["StringNode"]] = None) -> None`**

  Initializes a node with either a string or a nested list of nodes.

  - **Parameters:**
    - `value` (`Union[str, List["StringNode"]]`): The value to initialize the node with. Can be a string, a list of `StringNode` objects, or `None`.

  - **Raises:**
    - `ValueError`: If the value is not a string, a list of `StringNode`, or `None`.

- **`is_leaf(self) -> bool`**

  Checks if this node is a leaf node (contains a string).

  - **Returns:**
    - `bool`: `True` if the node is a leaf node, `False` otherwise.

- **`add(self, child: "StringNode") -> None`**

  Adds a child node to this node.

  - **Parameters:**
    - `child` (`StringNode`): The child node to add.

  - **Raises:**
    - `TypeError`: If the node is a leaf node (cannot add children to a leaf node).

- **`remove(self, child: "StringNode") -> None`**

  Removes a child node.

  - **Parameters:**
    - `child` (`StringNode`): The child node to remove.

  - **Raises:**
    - `TypeError`: If the node is a leaf node (cannot remove children from a leaf node).

- **`__repr__(self) -> str`**

  Returns a string representation of the node.

  - **Returns:**
    - `str`: A string representation of the node.

- **`__iter__(self) -> Iterator["StringNode"]`**

  Iterates over children if this is not a leaf node.

  - **Returns:**
    - `Iterator[StringNode]`: An iterator over the children nodes.

  - **Raises:**
    - `TypeError`: If the node is a leaf node (leaf nodes are not iterable).

### `MetaData`

A class for storing metadata information, using `pydantic` for data validation.

#### Attributes

- **`notes` (`StringNode | list | str`)**: Notes associated with the metadata. Can be a `StringNode`, a list, or a string.
- **`axes` (`str`)**: The axes of the metadata.
- **`TimeIncrement` (`float`)**: The time increment value.
- **`TimeIncrementUnit` (`Literal["s", "ms"]`)**: The unit of the time increment. Can be either "s" (seconds) or "ms" (milliseconds).
- **`PhysicalSizeX` (`float`)**: The physical size in the X dimension.
- **`PhysicalSizeXUnit` (`Literal["nm", "m"]`)**: The unit of the physical size in the X dimension. Can be either "nm" (nanometers) or "m" (meters).
- **`PhysicalSizeY` (`float`)**: The physical size in the Y dimension.
- **`PhysicalSizeYUnit` (`Literal["nm", "m"]`)**: The unit of the physical size in the Y dimension. Can be either "nm" (nanometers) or "m" (meters).

#### Methods

- **`__post_init__(self)`**

  Post-initialization method to convert the `notes` attribute to a `StringNode` if it is a list or a string.

#### Class Configuration

- **`Config`**:
  - **`arbitrary_types_allowed` (`bool`)**: Allows arbitrary types to be used in the model.

## Functions

### `to_string_node(nested: Union[str, List]) -> StringNode`

Converts a nested list of strings into a `StringNode` structure.

- **Parameters:**
  - `nested` (`Union[str, List]`): A string or a nested list of strings.

- **Returns:**
  - `StringNode`: A `StringNode` representing the nested structure.

- **Raises:**
  - `ValueError`: If the input is not a string or a nested list of strings.

---

This documentation provides an overview of the `metadata.py` module, detailing the classes, methods, and functions it contains. The `StringNode` class is particularly useful for creating hierarchical string structures, while the `MetaData` class is designed for storing and validating metadata information.