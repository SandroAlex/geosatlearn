import numpy as np
from shapely.geometry import Polygon


def create_area_name_from_geometry(geometry: Polygon) -> str:
    """
    Create an area name from the geometry using bounding box.
    """

    # Bounding box for this geometry.
    minx, miny, maxx, maxy = geometry.bounds

    # Create a unique area name based on the bounding box coordinates.
    if np.isclose(minx, maxx) or np.isclose(miny, maxy):
        raise ValueError(">>> Invalid geometry: bounding box has zero area!")
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise ValueError(
            ">>> Invalid geometry: bounding box coordinates out of bounds!"
        )

    # Area name based on the bounding box coordinates.
    name: str = f"minx_{minx:.6f}_miny_{miny:.6f}_maxx_{maxx:.6f}_maxy_{maxy:.6f}"

    return name
