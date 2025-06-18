"""
This module contains utility functions for working with time series.
"""

# Initial imports.
from typing import List

import xarray as xr


def add_vegetation_index_as_new_band(
    ds: xr.Dataset,
    satellite_code: str = "s2sr",
    band1_name: str = "nir",
    band2_name: str = "red",
    index_name: str = "ndvi",
) -> xr.Dataset:
    """
    Add a vegetation index (e.g., NDVI) as a new band to a satellite DataArray
    in an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset containing satellite bands.
    satellite_code : str, optional
        Name of the satellite variable in the Dataset (default is "s2sr").
    band1_name : str, optional
        Name of the first band (e.g., "nir") for index calculation.
    band2_name : str, optional
        Name of the second band (e.g., "red") for index calculation.
    index_name : str, optional
        Name of the new vegetation index band (default is "ndvi").

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the vegetation index added as a new band
        in the specified satellite variable.

    Examples
    --------
    >>> ds_with_ndvi = add_vegetation_index_as_new_band(ds, "s2sr", "nir", "red", "ndvi")
    >>> ds_with_ndvi["s2sr"].sel(band_s2sr="ndvi")
    """

    # Response dataset.
    ds_output: xr.Dataset = ds.copy(deep=True)

    # Calcule vindex.
    band1: xr.DataArray = ds_output[satellite_code].sel(
        {f"band_{satellite_code}": band1_name}
    )
    band2: xr.DataArray = ds_output[satellite_code].sel(
        {f"band_{satellite_code}": band2_name}
    )
    vindex: xr.DataArray = (band1 - band2) / (band1 + band2)

    # Add NDVI as a new band in the dataset.
    bands_names: List[str] = ds_output[satellite_code][
        f"band_{satellite_code}"
    ].values.tolist()
    bands_names: List[str] = bands_names + [index_name]

    # Original data.
    da: xr.DataArray = ds_output[satellite_code]

    # Create a new DataArray with the NDVI band.
    kwargs = {f"band_{satellite_code}": [index_name]}
    vindex: xr.DataArray = vindex.expand_dims(**kwargs)

    # Concatenate the original data with the new NDVI band.
    objs: List[xr.DataArray] = [da, vindex]
    da: xr.DataArray = xr.concat(objs=objs, dim=f"band_{satellite_code}")

    # Update the dataset with the new DataArray.
    ds_output = ds_output.drop_vars(
        [f"{satellite_code}", f"band_{satellite_code}", f"time_{satellite_code}"]
    )
    ds_output[satellite_code] = da

    return ds_output
