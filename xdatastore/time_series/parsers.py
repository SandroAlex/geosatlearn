"""
This module provides tools to convert geospatial long tables into xarray
datasets, organized by satellite, time, and bands. The main class,
`ParserLongTableAgriGEELite`, processes a GeoDataFrame with time series data
and outputs xarray datasets for each area, saving them in Zarr or NetCDF
format.
"""

# Initial imports.
import os
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import Polygon
from tqdm.auto import tqdm

from xdatastore.geometries.utils import create_area_name_from_geometry


class LongTableAgriGEELiteToXarrayDataset:
    """
    Parser to transform long tables into xarray datasets.

    This class converts geospatial data organized in long tabular format
    into xarray Dataset format, organized by satellite, time, and bands.

    Parameters
    ----------
    base_folder : str, optional
        Base folder where datasets will be saved.
    config_crosswalk_bands : Dict[str, list], optional
        Dictionary mapping satellite codes to their bands.
    verbose : bool, optional
        If True, display progress bars.
    save_format : str, optional
        Format to save datasets ('zarr' or 'netcdf').
    """

    def __init__(
        self,
        base_folder: str = "s3://agrilearn-xarray-datasets",
        config_crosswalk_bands: Dict[str, List[str]] = {
            "s1d": ["vv", "vh"],
            "s2sr": [
                "blue",
                "green",
                "red",
                "re1",
                "re2",
                "re3",
                "nir",
                "re4",
                "swir1",
                "swir2",
            ],
            "l8sr": ["blue", "green", "red", "nir", "swir1", "swir2"],
            "l9sr": ["blue", "green", "red", "nir", "swir1", "swir2"],
        },
        verbose: bool = True,
        save_format: str = "zarr",
    ) -> None:
        """
        Initialize the parser for long tables to xarray datasets.

        Parameters
        ----------
        base_folder : str
            Base folder where datasets will be saved.
        config_crosswalk_bands : Dict[str, list], optional
            Dictionary mapping satellite codes to their bands.
        verbose : bool, optional
            If True, display progress bars.
        save_format : str, optional
            Format to save datasets ('zarr' or 'netcdf').
        """
        self.base_folder: str = base_folder
        self.config_crosswalk_bands: Dict[str, List[str]] = config_crosswalk_bands
        self.verbose: bool = verbose
        self.save_format: str = save_format

    def run(
        self,
        input_gdf: gpd.GeoDataFrame,
        start_date_column_name: str = "start_date",
        satellites_codes: List[str] = ["s1d", "s2sr", "l8sr", "l9sr"],
    ) -> gpd.GeoDataFrame:
        """
        Process input data and create xarray datasets for each area.

        This method iterates over all areas in the input GeoDataFrame and
        creates an xarray dataset for each, processing all bands specified
        for each satellite code.

        Parameters
        ----------
        input_gdf : gpd.GeoDataFrame
            GeoDataFrame containing input data with time series and geometries.
        satellites_codes : List[str], optional
            List of satellite codes to process.
            Default is ["s1d", "s2sr", "l8sr", "l9sr"].

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with area geometries and paths to datasets.
        """

        output_gdf: gpd.GeoDataFrame = input_gdf.copy(deep=True)

        output_fields_iterator = tqdm(
            range(len(output_gdf)), desc=f">>> Processing input field areas ..."
        )

        for intidx in output_fields_iterator:

            input_field_gdf: gpd.GeoDataFrame = output_gdf.iloc[[intidx]]
            geometry: Polygon = input_field_gdf.geometry.values[0]
            start_date_year: str = str(
                input_field_gdf[start_date_column_name].dt.year.values[0]
            )
            df_sits_bands: Dict[str, pd.DataFrame] = self._create_time_series_dict(
                input_field_gdf=input_field_gdf, satellites_codes=satellites_codes
            )
            ds: xr.Dataset = self._create_xarray_dataset_from_time_series_dict(
                df_sits_bands=df_sits_bands
            )
            geometry_wkt: str = geometry.wkt
            ds.attrs["geometry_wkt"] = geometry_wkt
            dataset_full_path: str = self._save_dataset(
                ds=ds, geometry=geometry, start_date_year=start_date_year
            )
            output_gdf.at[intidx, "dataset_full_path"] = dataset_full_path

        return output_gdf

    def _create_time_series_dict(
        self, input_field_gdf: gpd.GeoDataFrame, satellites_codes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Create time series dictionary for each satellite in a single area.

        Parameters
        ----------
        input_field_gdf : gpd.GeoDataFrame
            GeoDataFrame containing data for a single area.
        satellites_codes : List[str]
            List of satellite codes to process.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping satellite codes to DataFrames with band time
            series.
        """
        df_sits_bands: Dict[str, pd.DataFrame] = {}
        for satellite_code in satellites_codes:
            df_bands: pd.DataFrame = self._get_satellite_bands(
                input_field_gdf=input_field_gdf, satellite_code=satellite_code
            )
            df_sits_bands.update({satellite_code: df_bands})
        return df_sits_bands

    def _create_xarray_dataset_from_time_series_dict(
        self,
        df_sits_bands: Dict[str, pd.DataFrame],
    ) -> xr.Dataset:
        """
        Create xarray dataset from time series dictionary.

        Converts each time series DataFrame (for each satellite) into an xarray
        DataArray, and combines all into a single Dataset.

        Parameters
        ----------
        df_sits_bands : Dict[str, pd.DataFrame]
            Dictionary mapping satellite codes to DataFrames with band time
            series.

        Returns
        -------
        xr.Dataset
            xarray Dataset containing data from all satellites.
        """
        data_dict: Dict[str, xr.DataArray] = {}
        for satellite_code, df_bands in df_sits_bands.items():
            df_bands_doy_dropped: pd.DataFrame = df_bands.drop(columns=["doy"])
            da_bands: xr.DataArray = xr.DataArray(
                data=df_bands_doy_dropped.values,
                dims=[f"time_{satellite_code}", f"band_{satellite_code}"],
                coords={
                    f"time_{satellite_code}": df_bands_doy_dropped.index.values,
                    f"band_{satellite_code}": df_bands_doy_dropped.columns.tolist(),
                },
            )
            data_dict[satellite_code] = da_bands
        ds: xr.Dataset = xr.Dataset(data_vars=data_dict)
        return ds

    def _get_satellite_bands(
        self, input_field_gdf: gpd.GeoDataFrame, satellite_code: str
    ) -> pd.DataFrame:
        """
        Extract bands for a given satellite code and specific area.

        This method processes the data from the long table to create a
        time-indexed series for each band of the specified satellite.

        Parameters
        ----------
        input_field_gdf : gpd.GeoDataFrame
            GeoDataFrame containing data for a single area.
        satellite_code : str
            Satellite code for which bands will be extracted.

        Returns
        -------
        pd.DataFrame
            DataFrame with time index containing all bands for the specified
            satellite.
        """
        columns: List[str] = [
            col for col in input_field_gdf.columns if satellite_code in col.lower()
        ]
        observations_col: List[str] = [col for col in columns if "observations" in col]
        observations_n: int = int(input_field_gdf[observations_col].values.squeeze())
        doys_columns: List[str] = [col for col in columns if "doy" in col.lower()]
        doys_values: np.ndarray = input_field_gdf[doys_columns].values.flatten()
        doys_nonzeros_idxs: np.ndarray = doys_values != 0
        doys_values: np.ndarray = doys_values[doys_nonzeros_idxs]
        year_changes: np.ndarray = np.diff(doys_values, prepend=0) < 0
        start_year: int = int(input_field_gdf["start_date"].dt.year.values[0])
        end_year: int = int(input_field_gdf["end_date"].dt.year.values[0])
        year: int = start_year
        timestamps: List[pd.Timestamp] = []
        for year_change, doy_value in zip(year_changes, doys_values):
            if year_change:
                year += 1
            timestamp: pd.Timestamp = pd.Timestamp(
                year=year, month=1, day=1
            ) + pd.to_timedelta(int(doy_value) - 1, unit="D")
            timestamps.append(timestamp)
        bands_names: List[str] = self.config_crosswalk_bands[satellite_code]
        df_tmp: List[pd.DataFrame] = []
        for band_name in bands_names:
            bands_cols: List[str] = [
                col for col in columns if f"{band_name}" in col.lower()
            ]
            if not bands_cols:
                continue
            band_values: np.ndarray = input_field_gdf[bands_cols].values.flatten()
            band_values: np.ndarray = band_values[doys_nonzeros_idxs]
            df_band: pd.DataFrame = pd.DataFrame(
                data={
                    "doy": doys_values,
                    "timestamp": timestamps,
                    f"{band_name}": band_values,
                }
            )
            df_band.set_index("timestamp", inplace=True)
            df_tmp.append(df_band)
        df_bands: pd.DataFrame = df_tmp[0]
        for df_band in df_tmp[1:]:
            df_bands = df_bands.merge(df_band, how="inner", on=["doy", "timestamp"])
        assert observations_n == len(df_bands), (
            f">>> Number of observations ({observations_n}) does not match number "
            f"of timestamps ({len(df_bands)})!"
        )
        return df_bands

    def _save_dataset(
        self, ds: xr.Dataset, geometry: Polygon, start_date_year: str
    ) -> str:
        """
        Save the xarray Dataset to disk or S3 in Zarr format.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to save.
        geometry : shapely.Polygon
            Geometry of the area (used for naming).
        start_date_year : str
            Year string for folder structure.

        Returns
        -------
        str
            Full path where the dataset was saved.
        """
        area_name: str = create_area_name_from_geometry(geometry=geometry)

        # Names of satellites present in dataset.
        satellites_codes: List[str] = list(ds.data_vars.keys())

        # Join all codes in the same string.
        satellites_names: str = "-".join(satellites_codes)

        # Patern example:
        # s3://agrilearn-xarray-datasets/minx_-46.208007_miny_-17.056277_maxx_-46.204046_maxy_-17.052840/2023/time-series/s2sr-s1d-l8sr-l9sr-raw.zarr/
        dataset_full_path: str = os.path.join(
            self.base_folder,
            f"{area_name}",
            f"{start_date_year}",
            "time-series",
            f"{satellites_names}-raw.{self.save_format}",
        )

        if self.verbose:
            print(f">>> Saving dataset to {dataset_full_path} ...")

        if self.save_format == "zarr":
            ds.to_zarr(dataset_full_path, mode="w", consolidated=True)

        if self.save_format == "netcdf":
            ds.to_netcdf(dataset_full_path, mode="w")

        return dataset_full_path
