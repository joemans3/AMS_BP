import numpy as np


def generate_nup96_positions(
    ring_diameter: float = 107.0,
    molecule_spacing: float = 12.0,
    ring_spacing: float = 50.0,
) -> np.ndarray:
    """
    Generate the 3D coordinates of Nup96 proteins in the nuclear pore complex.

    Parameters:
    -----------
    ring_diameter : float
        Diameter of the main ring in nanometers (default: 107.0 nm)
    molecule_spacing : float
        Distance between two Nup96 molecules within same section (default: 12.0 nm)
    ring_spacing : float
        Distance between nuclear and cytoplasmic rings in z-direction (default: 50.0 nm)

    Returns:
    --------
    numpy.ndarray
        Array of shape (32, 3) containing x, y, z coordinates for all Nup96 proteins in um
        First 32 coordinates are the main structural Nup96 (16 nuclear, 16 cytoplasmic)
    """

    # Initialize array to store coordinates
    coordinates = np.zeros((32, 3))
    ring_radius = ring_diameter / 2

    # Generate positions for both main rings (nuclear and cytoplasmic)
    for ring in range(2):  # 0 = nuclear side, 1 = cytoplasmic side
        z = ring_spacing / 2 if ring == 0 else -ring_spacing / 2

        for octant in range(8):
            # Calculate base angle for this octant
            base_angle = octant * 2 * np.pi / 8

            # Calculate the center position of this octamer
            center_x = ring_radius * np.cos(base_angle)
            center_y = ring_radius * np.sin(base_angle)

            # Place two Nup96 molecules in each octant section
            for molecule in range(2):
                # Calculate the offset direction perpendicular to the radius
                perpendicular_angle = (
                    base_angle
                    + np.pi / 2
                    + molecule_spacing / 3 * (1 if molecule == 0 else -1)
                )

                # Offset from center position
                offset = (molecule_spacing / 3) * (-1 if molecule == 0 else 1) + (
                    molecule_spacing / 5
                ) * (-1 if ring == 0 else 1)

                # Calculate final x and y coordinates
                x = center_x + offset * np.cos(perpendicular_angle)
                y = center_y + offset * np.sin(perpendicular_angle)

                # Store coordinates
                idx = ring * 16 + octant * 2 + molecule
                coordinates[idx] = [x, y, z]
                # add 1 nm gitter
                coordinates[idx] += np.random.normal(0, 1, 3)

    return coordinates / 1000.0
