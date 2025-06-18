from typing import List

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from shapely import wkt
from shapely.geometry import Polygon


def apply_negative_buffer(
    geometry: Polygon,
    n_pixels_negative_buffer: int = 5,
    crs_linear: int = int(3857),
    pixel_resolution_linear: int = 10,
) -> Polygon:
    """
    Apply a negative buffer to the geometry.

    Parameters
    ----------
    geometry : Polygon
        The geometry to apply the buffer to.
    n_pixels_negative_buffer : int, optional, default is 5
        The number of pixels to buffer.
    crs_linear : int, optional
        The coordinate reference system to use for linear measurements (default is 3857).
    pixel_resolution_linear : int, optional
        The resolution of the pixels in the linear CRS (default is 10).

    Returns
    -------
    Polygon
        The buffered geometry.
    """

    geometry_buffered: Polygon = (
        gpd.GeoDataFrame(data={"geometry": [geometry]}, crs="EPSG:4326")
        .to_crs(epsg=crs_linear)
        .buffer((-1) * n_pixels_negative_buffer * pixel_resolution_linear)
        .to_crs(epsg=4326)
        .geometry[0]
    )

    return geometry_buffered


def add_input_area_mask(
    ds: xr.Dataset, mask_name: str = "input_field_mask"
) -> xr.Dataset:
    """
    Add an input area mask to the dataset.

    The input area mask is a boolean mask that indicates the area of interest
    based on the geometry provided in the dataset attributes.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the geometry in its attributes.
    mask_name : str, optional
        The name of the mask to be added to the dataset (default is
        "input_field_mask").

    Returns
    -------
    xr.Dataset
        The dataset with the input area mask added as a new data array.
    """

    # Initialize response.
    ds_output: xr.Dataset = ds.copy(deep=True)

    # Extract geometry.
    geometry_wkt: str = ds.attrs.get("geometry_wkt", None)
    geometry: Polygon = wkt.loads(geometry_wkt)

    # Create xarray data array with the geometry.
    input_field_mask: xr.DataArray = ds_output.s2sr.isel(time=0, band=0).rio.clip(
        geometries=[geometry.__geo_interface__], drop=False
    )
    input_field_mask: xr.DataArray = np.logical_not(np.isnan(input_field_mask))

    # Add to dataset.
    ds_output[mask_name] = input_field_mask

    return ds_output


def add_input_area_mask_buffered(
    ds: xr.Dataset,
    n_pixels_negative_buffer: int = 5,
    mask_name: str = "input_field_mask",
    mask_buffered_name: str = "input_field_buffered_mask",
) -> xr.Dataset:
    """
    Add an input area mask to the dataset with a negative buffer.

    The input area mask is a boolean mask that indicates the area of interest
    based on the geometry provided in the dataset attributes, with a negative
    buffer applied to the geometry.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the geometry in its attributes.
    n_pixels_negative_buffer : int, optional
        The number of pixels to apply as a negative buffer to the geometry
        (default is 5).
    mask_name : str, optional
        The name of the mask to be added to the dataset (default is
        "input_field_mask").
    mask_buffered_name : str, optional
        The name of the buffered mask to be added to the dataset
        (default is "input_field_buffered_mask").

    Returns
    -------
    xr.Dataset
        The dataset with the input area mask and the buffered mask added as new
        data arrays.
    """

    # Initialize response.
    ds_output: xr.Dataset = ds.copy(deep=True)

    # Extract geometry.
    geometry_wkt: str = ds_output.attrs.get("geometry_wkt", None)
    geometry: Polygon = wkt.loads(geometry_wkt)

    # Add input area mask.
    ds_output: xr.Dataset = add_input_area_mask(ds=ds_output, mask_name=mask_name)

    # Apply negative buffer to the geometry.
    geometry_buffered: Polygon = apply_negative_buffer(
        geometry=geometry, n_pixels_negative_buffer=n_pixels_negative_buffer
    )

    # Add buffered geometry to the dataset as a xarray data array.
    input_field_buffered_mask: xr.DataArray = ds_output.s2sr.isel(
        time=0, band=0
    ).rio.clip(geometries=[geometry_buffered.__geo_interface__], drop=False)
    input_field_buffered_mask: xr.DataArray = np.logical_not(
        np.isnan(input_field_buffered_mask)
    )
    ds_output[mask_buffered_name] = input_field_buffered_mask

    return ds_output


def create_mask_of_valid_pixels(
    ds: xr.Dataset,
    n_pixels_negative_buffer: int = 5,
    scl_valid_classes: List[int] = [4, 5, 6, 7, 9],
    mask_name: str = "input_field_mask",
    mask_buffered_name: str = "input_field_buffered_mask",
    valid_pixels_name: str = "valid_pixels_mask",
) -> xr.Dataset:
    """
    Create a mask of valid pixels in the dataset.
    The valid pixels are those that are inside the input area mask and
    belong to the specified SCL valid classes.
    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the geometry in its attributes and the
        SCL band.
    n_pixels_negative_buffer : int, optional
        The number of pixels to apply as a negative buffer to the geometry
        (default is 5).
    scl_valid_classes : List[int], optional
        The list of valid SCL classes to consider as valid pixels
        (default is [4, 5, 6, 7, 9]).
    mask_name : str, optional
        The name of the input area mask to be added to the dataset
        (default is "input_field_mask").
    mask_buffered_name : str, optional
        The name of the buffered mask to be added to the dataset
        (default is "input_field_buffered_mask").
    valid_pixels_name : str, optional
        The name of the valid pixels mask to be added to the dataset
        (default is "valid_pixels_mask").

    Returns
    -------
    xr.Dataset
        The dataset with the valid pixels mask added as a new data array.
    """

    # Initialize response.
    ds_output: xr.Dataset = ds.copy(deep=True)

    # Add buffered input area mask to the dataset.
    ds_output: xr.Dataset = add_input_area_mask_buffered(
        ds=ds,
        n_pixels_negative_buffer=n_pixels_negative_buffer,
        mask_name=mask_name,
        mask_buffered_name=mask_buffered_name,
    )

    # Mask pixels for scl valid classes.
    mask_scl: xr.DataArray = ds_output.s2sr.sel(band="scl").isin(scl_valid_classes)

    # Valid pixels inside input area.
    final_mask: xr.DataArray = np.logical_and(ds_output[mask_buffered_name], mask_scl)

    # Fill response.
    ds_output[valid_pixels_name] = final_mask

    return ds_output
