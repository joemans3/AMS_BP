# Cell Configuration Help

This section defines the geometric model of the cell used in the simulation. Each cell type has specific parameters that describe its spatial extent and shape.

## Fields

### `cell_type`
- **Type**: `String`
- **Options**: `"RectangularCell"`, `"SphericalCell"`, `"OvoidCell"`, `"RodCell"`, `"BuddingCell"`
- **Description**: Specifies the type of cell geometry to simulate.

---

## Parameters by Cell Type

### **SphericalCell**
- `center`: `[x, y, z]` — 3D coordinates of the center of the sphere.
- `radius`: `float` — Radius of the sphere in micrometers.

### **RodCell**
- `center`: `[x, y, z]` — 3D center of the rod.
- `direction`: `[dx, dy, dz]` — Direction vector of the rod’s long axis (will be normalized).
- `height`: `float` — Length of the rod in micrometers.
- `radius`: `float` — Radius of the rod in micrometers.

### **RectangularCell**
- `bounds`: `[xmin, xmax, ymin, ymax, zmin, zmax]` — Bounds of the cell volume in micrometers. Useful for simulating flat, box-like geometries.

### **OvoidCell**
- `center`: `[x, y, z]` — 3D center of the ovoid.
- `xradius`: `float` — Radius along the x-axis.
- `yradius`: `float` — Radius along the y-axis.
- `zradius`: `float` — Radius along the z-axis.

### **BuddingCell**
- *Not currently implemented in the UI.*

---

## Notes
- The selected `cell_type` determines which fields are editable.
- All units are in micrometers (`μm`).
- The coordinate system is 3D and consistent with the z-position defined in the experiment setup.
- These values define the spatial domain for molecule simulation, diffusion, and tracking.

