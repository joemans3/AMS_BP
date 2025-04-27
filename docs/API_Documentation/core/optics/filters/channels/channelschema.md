# Module: `channelsschema.py`

This module defines a Pydantic model for representing a collection of channels, including their associated filter sets, splitting efficiencies, and names.

## Classes

### `Channels`

A Pydantic model representing a collection of channels.

#### Attributes

- **`filtersets`**: `List[FilterSet]`
  - A list of `FilterSet` objects, each representing a set of filters for a channel.
  
- **`num_channels`**: `int`
  - The number of channels. This value must match the length of the `filtersets` and `splitting_efficiency` lists.
  
- **`splitting_efficiency`**: `List[float]`
  - A list of floats representing the splitting efficiency for each channel. The length of this list must match the `num_channels`.
  
- **`names`**: `List[str]`
  - A list of strings representing the names of the channels.

#### Validation

- **`num_channels`**:
  - The `num_channels` attribute is validated to ensure that it matches the length of the `filtersets` and `splitting_efficiency` lists. If the lengths do not match, a `ValueError` is raised.

#### Example

```python
from channelsschema import Channels
from ....optics.filters.filters import FilterSet

# Example usage
channels = Channels(
    filtersets=[FilterSet(...), FilterSet(...)],
    num_channels=2,
    splitting_efficiency=[0.8, 0.9],
    names=["Channel 1", "Channel 2"]
)
```

#### Raises

- **`ValueError`**:
  - If `num_channels` does not match the length of `filtersets`.
  - If `num_channels` does not match the length of `splitting_efficiency`.

---