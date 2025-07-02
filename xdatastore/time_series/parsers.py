# Initial imports.
import os
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import Polygon
from shapely.wkt import loads
from tqdm.auto import tqdm

from xdatastore.geometries.utils import create_area_name_from_geometry


class WideTableAgriGEELiteToXarrayDataset:
    """
    Parser to transform wide tables into xarray datasets.

    This class converts geospatial data organized in wide tabular format
    into xarray Dataset format, organized by satellite, time, and bands.
    """

    def __init__(
        self,
        base_folder: str,
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
        save_format: str = "zarr",
        verbose: bool = True,
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
        self.save_format: str = save_format
        self.verbose: bool = verbose

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
        bands_column_names: List[str] = self.config_crosswalk_bands[satellite_code]
        df_tmp: List[pd.DataFrame] = []
        for band_name in bands_column_names:
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


class LongTableAgriGEELiteToXarrayDataset:
    """
    Parser to transform long tables into xarray datasets.

    This class converts geospatial data organized in long tabular format
    into xarray Dataset format, organized by satellite, time, and bands.
    """

    def __init__(
        self,
        base_folder: str,
        default_crs: int = 4326,
        satellite_code: str = "mod09g",
        save_format: str = "zarr",
        verbose: bool = True,
    ) -> None:
        """
        Initialize the parser for long tables to xarray datasets.

        Parameters
        ----------
        base_folder : str
            Base folder where datasets will be saved.
        default_crs : int, optional
            Default CRS for output geometries (default is 4326).
        satellite_code : str, optional
            Satellite code for the dataset (default is "mod250").
        save_format : str, optional
            Format to save datasets ('zarr' or 'netcdf').
        verbose : bool, optional
            If True, display progress bars and messages.
        """

        # Main parameters.
        self.base_folder: str = base_folder
        self.default_crs: int = default_crs
        self.satellite_code: str = satellite_code
        self.save_format: str = save_format
        self.verbose: bool = verbose

    def run(
        self,
        input_geometries_gdf: gpd.GeoDataFrame,
        input_time_series_df: pd.DataFrame,
        start_date_column_name: str = "start_date",
        end_date_column_name: str = "end_date",
        bands_column_names: List[str] = ["red", "nir"],
        ndvi_column_name: str = "ndvi",
        timestamp_column_name: str = "timestamp",
        index_column_name_for_geometries: str = "00_indexnum",
        index_column_name_for_time_series: str = "indexnum",
    ) -> gpd.GeoDataFrame:
        """
        Process input geometries and time series, and create xarray datasets
        for each area.

        Parameters
        ----------
        input_geometries_gdf : gpd.GeoDataFrame
            GeoDataFrame containing area geometries and date columns.
        input_time_series_df : pd.DataFrame
            DataFrame containing time series data for all areas.
        start_date_column_name : str, optional
            Name of the start date column (default is "start_date").
        end_date_column_name : str, optional
            Name of the end date column (default is "end_date").
        bands_column_names : List[str], optional
            List of band column names (default is ["red", "nir"]).
        ndvi_column_name : str, optional
            Name of the NDVI column (default is "ndvi").
        timestamp_column_name : str, optional
            Name of the timestamp column (default is "timestamp").
        index_column_name_for_geometries : str, optional
            Name of the index column in geometries GeoDataFrame
            (default is "00_indexnum").
        index_column_name_for_time_series : str, optional
            Name of the index column in time series DataFrame
            (default is "indexnum").

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with area geometries and dataset paths.
        """
        # Initialize output GeoDataFrame with geometry and date columns.
        output_gdf: gpd.GeoDataFrame = input_geometries_gdf[
            [
                "geometry",
                start_date_column_name,
                end_date_column_name,
                index_column_name_for_geometries,
            ]
        ].copy(deep=True)
        output_gdf = output_gdf.to_crs(epsg=self.default_crs)

        # Loop over all areas in the input GeoDataFrame.
        for intidx in tqdm(
            range(len(output_gdf)), desc=f">>> Processing input field areas ..."
        ):

            # Input area (talhao).
            input_field_gdf: gpd.GeoDataFrame = output_gdf.iloc[[intidx]]

            # Grab geometry.
            geometry: Polygon = input_field_gdf["geometry"].values[0]

            # Start date for this geometry.
            start_date_year: str = str(
                input_field_gdf[start_date_column_name].dt.year.values[0]
            )

            # Grab time series for this geometry.
            idx: int = int(input_field_gdf[index_column_name_for_geometries].values[0])
            mask: pd.Series = (
                input_time_series_df[index_column_name_for_time_series].astype(int)
                == idx
            )
            time_series: pd.DataFrame = input_time_series_df[mask].copy(deep=True)

            # Create xarray Dataset from the time series.
            ds: xr.Dataset = self._create_xarray_dataset(
                time_series=time_series,
                timestamp_column_name=timestamp_column_name,
                bands_column_names=bands_column_names,
                ndvi_column_name=ndvi_column_name,
            )

            # Save geometry in wkt format as an attribute of dataset.
            geometry_wkt: str = geometry.wkt
            ds.attrs["geometry"] = geometry_wkt

            # Calculate area of the geometry in hectares.
            area: float = self._calculate_area(input_field_gdf=input_field_gdf)

            # Save area as an attribute of this dataset.
            ds.attrs["surface_area_ha"] = area

            # Save the dataset to disk or S3.
            save_path: str = self._save_dataset(
                ds=ds, geometry=geometry, start_date_year=start_date_year
            )
            output_gdf.at[intidx, "dataset_full_path"] = save_path

            if self.verbose:
                print(">>> Finished processing area!\n")

        # Return the output GeoDataFrame with dataset paths.
        return output_gdf

    def _create_xarray_dataset(
        self,
        time_series: pd.DataFrame,
        timestamp_column_name: str,
        bands_column_names: List[str],
        ndvi_column_name: str,
    ) -> xr.Dataset:
        """
        Create an xarray Dataset from a time series DataFrame.

        Parameters
        ----------
        time_series : pd.DataFrame
            DataFrame containing time series data for one area.
        timestamp_column_name : str
            Name of the timestamp column.
        bands_column_names : List[str]
            List of band column names.
        ndvi_column_name : str
            Name of the NDVI column.

        Returns
        -------
        xr.Dataset
            xarray Dataset containing bands and NDVI for the area.
        """
        # Extract timestamps and bands values. Also extract NDVI values.
        timestamps: np.array = pd.to_datetime(
            time_series[timestamp_column_name], format="%Y-%m-%d"
        ).values
        bands_values: np.array = time_series[bands_column_names].values
        ndvi_values: np.array = time_series[ndvi_column_name].values

        # Create xarray DataArray for bands.
        da_bands: xr.DataArray = xr.DataArray(
            data=bands_values,
            dims=[f"time_{self.satellite_code}", f"band_{self.satellite_code}"],
            coords={
                f"time_{self.satellite_code}": timestamps,
                f"band_{self.satellite_code}": bands_column_names,
            },
            name=self.satellite_code,
        )

        # Create xarray DataArray for NDVI.
        da_ndvi: xr.DataArray = xr.DataArray(
            data=ndvi_values,
            dims=[f"time_{self.satellite_code}"],
            coords={f"time_{self.satellite_code}": timestamps},
            name=f"ndvi_{self.satellite_code}",
        )

        # Join both DataArrays into a Dataset.
        ds: xr.Dataset = xr.Dataset({da_bands.name: da_bands, da_ndvi.name: da_ndvi})

        return ds

    def _calculate_area(
        self,
        input_field_gdf: gpd.GeoDataFrame,
        epsg_area: int = 3857,
        scaling_factor_area: float = 1.0 / 10000.0,
    ) -> float:
        """
        Calculate the area of the input field in hectares.

        Parameters
        ----------
        input_field_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing the field geometry.
        epsg_area : int, optional
            EPSG code for area calculation (default is 3857).
        scaling_factor_area : float, optional
            Factor to convert area to hectares.

        Returns
        -------
        float
            Area in hectares.
        """
        area: float = float(
            scaling_factor_area * input_field_gdf.to_crs(epsg_area).area.values[0]
        )

        if self.verbose:
            print(f">>> Area of the input field: {area:.2f} hectares ...")

        return area

    def _save_dataset(
        self, ds: xr.Dataset, geometry: Polygon, start_date_year: str
    ) -> str:
        """
        Save the xarray Dataset to disk or S3 in Zarr or NetCDF format.

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
        # Area name based on geometry bounding box.
        area_name: str = create_area_name_from_geometry(geometry=geometry)

        # Patern example:
        # s3://agrilearn-xarray-datasets/minx_-57.796715_miny_-13.165000_maxx_-57.784700_maxy_-13.141965/2023/time-series/mod250-raw.zarr
        dataset_full_path: str = os.path.join(
            self.base_folder,
            f"{area_name}",
            f"{start_date_year}",
            "time-series",
            f"{self.satellite_code}-raw.{self.save_format}",
        )

        if self.verbose:
            print(f">>> Saving dataset to {dataset_full_path} ...")

        if self.save_format == "zarr":
            ds.to_zarr(dataset_full_path, mode="w", consolidated=True)

        if self.save_format == "netcdf":
            ds.to_netcdf(dataset_full_path, mode="w")

        return dataset_full_path


class HarmonizedS2L8L9ToXarrayDataset:
    """
    Parser to transform harmonized S2, L8, and L9 tables into xarray datasets.

    This class converts harmonized geospatial data from S2, L8, and L9 sensors,
    organized in tabular format, into xarray Dataset format, organized by
    satellite, time, and bands.
    """

    def __init__(
        self,
        base_folder: str = "s3://agrilearn-xarray-datasets",
        satellite_code: str = "s2l8l9",
        save_format: str = "zarr",
        verbose: bool = True,
    ) -> None:
        """
        Initialize the parser for harmonized S2/L8/L9 tables to xarray datasets.

        Parameters
        ----------
        base_folder : str, optional
            Base folder where datasets will be saved.
        satellite_code : str, optional
            Satellite code for the dataset (default is "s2l8l9").
        save_format : str, optional
            Format to save datasets ('zarr' or 'netcdf').
        verbose : bool, optional
            If True, display progress bars and messages.
        """
        # Main parameters.
        self.base_folder: str = base_folder
        self.satellite_code: str = satellite_code
        self.save_format: str = save_format
        self.verbose: bool = verbose

    def run(
        self,
        input_gdf: gpd.GeoDataFrame,
        timestamp_column_name: str = "date",
        bands_column_names: List[str] = [
            "Blue_adj",
            "Green_adj",
            "Red_adj",
            "NIR_adj",
            "SWIR1_adj",
            "SWIR2_adj",
        ],
        ndvi_column_name: str = "NDVI_adj",
    ) -> gpd.GeoDataFrame:
        """
        Process input harmonized data and create xarray datasets for each area.

        This method iterates over all unique geometries in the input
        GeoDataFrame and creates an xarray dataset for each, processing all
        specified bands.

        Parameters
        ----------
        input_gdf : gpd.GeoDataFrame
            GeoDataFrame containing harmonized input data with time series and
            geometries.
        timestamp_column_name : str, optional
            Name of the timestamp column (default is "date").
        bands_column_names : List[str], optional
            List of band column names to process.
        ndvi_column_name : str, optional
            Name of the NDVI column (default is "NDVI_adj").

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with area geometries and paths to saved datasets.
        """
        # Initialize response as an void geodataframe.
        output_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()

        # List all available geometries.
        geometries: List[Polygon] = self._list_all_available_geometries(
            input_gdf=input_gdf
        )

        # Loop over all geoemetries and create a mask for each geometry.
        for geometry in geometries:

            # Create a mask for the current geometry.
            mask: pd.Series = input_gdf.geometry == geometry.wkt

            # Filter the input GeoDataFrame using the mask.
            filtered_gdf: gpd.GeoDataFrame = input_gdf[mask]

            # If the filtered GeoDataFrame is empty, skip to the next geometry.
            if filtered_gdf.empty:
                continue

            # Remove timestamps with nans.
            filtered_gdf = filtered_gdf.dropna(how="any")

            # Start date year for the current geometry.
            start_date_year: str = str(
                pd.to_datetime(filtered_gdf[timestamp_column_name], format="%Y-%m-%d")
                .min()
                .year
            )

            # Create the xarray Dataset for the current geometry.
            ds: xr.Dataset = self._create_xarray_dataset(
                time_series=filtered_gdf,
                timestamp_column_name=timestamp_column_name,
                bands_column_names=bands_column_names,
                ndvi_column_name=ndvi_column_name,
            )

            # Calculate area of geometry in hectares.
            area: float = self._calculate_area(geometry=geometry)

            # Save area as an attribute in the Dataset.
            ds.attrs["surface_area_ha"] = area

            # Save geometry as an attribute in the Dataset.
            ds.attrs["geometry_wkt"] = geometry.wkt

            # Complete path.
            save_path: str = self._save_dataset(
                ds=ds, geometry=geometry, start_date_year=start_date_year
            )

            # Add geometry and save path to the output GeoDataFrame.
            output_gdf: gpd.GeoDataFrame = pd.concat(
                [
                    output_gdf,
                    gpd.GeoDataFrame(
                        {
                            "geometry": [geometry],
                            "save_path": [save_path],
                            "start_date_year": [start_date_year],
                            "surface_area_ha": [area],
                        }
                    ),
                ]
            )

            if self.verbose:
                print(">>> Finished!\n")

        return output_gdf

    def _list_all_available_geometries(
        self, input_gdf: gpd.GeoDataFrame
    ) -> List[Polygon]:
        """
        List all unique geometries in the input GeoDataFrame.

        Parameters
        ----------
        input_gdf : gpd.GeoDataFrame
            The input GeoDataFrame containing geometries.

        Returns
        -------
        List[Polygon]
            A list of unique geometries as Shapely Polygon objects.
        """
        # Extract unique geometries from the input GeoDataFrame.
        geometries: List[str] = input_gdf.geometry.unique().tolist()

        # Convert the geometries to Shapely Polygon objects.
        geometries: List[Polygon] = [loads(geom) for geom in geometries]

        return geometries

    def _create_xarray_dataset(
        self,
        time_series: pd.DataFrame,
        timestamp_column_name: str,
        bands_column_names: List[str],
        ndvi_column_name: str,
    ) -> xr.Dataset:
        """
        Create an xarray Dataset from a time series DataFrame.

        Parameters
        ----------
        time_series : pd.DataFrame
            DataFrame containing time series data for one area.
        timestamp_column_name : str
            Name of the timestamp column.
        bands_column_names : List[str]
            List of band column names.
        ndvi_column_name : str
            Name of the NDVI column.

        Returns
        -------
        xr.Dataset
            xarray Dataset containing bands and NDVI for the area.
        """
        # Extract timestamps and bands values. Also extract NDVI values.
        timestamps: np.array = pd.to_datetime(
            time_series[timestamp_column_name], format="%Y-%m-%d"
        ).values
        bands_values: np.array = time_series[bands_column_names].values
        ndvi_values: np.array = time_series[ndvi_column_name].values

        # Create xarray DataArray for bands.
        da_bands: xr.DataArray = xr.DataArray(
            data=bands_values,
            dims=[f"time_{self.satellite_code}", f"band_{self.satellite_code}"],
            coords={
                f"time_{self.satellite_code}": timestamps,
                f"band_{self.satellite_code}": bands_column_names,
            },
            name=self.satellite_code,
        )

        # Create xarray DataArray for NDVI.
        da_ndvi: xr.DataArray = xr.DataArray(
            data=ndvi_values,
            dims=[f"time_{self.satellite_code}"],
            coords={f"time_{self.satellite_code}": timestamps},
            name=f"ndvi_{self.satellite_code}",
        )

        # Join both DataArrays into a Dataset.
        ds: xr.Dataset = xr.Dataset({da_bands.name: da_bands, da_ndvi.name: da_ndvi})

        return ds

    def _calculate_area(
        self,
        geometry: Polygon,
        epsg_area: int = 3857,
        scaling_factor_area: float = 1.0 / 10000.0,
    ) -> float:
        """
        Calculate the area of the input field in hectares.

        Parameters
        ----------
        geometry : Polygon
            The geometry of the input field.
        epsg_area : int, optional
            EPSG code for area calculation (default is 3857).
        scaling_factor_area : float, optional
            Factor to convert area to hectares.

        Returns
        -------
        float
            Area in hectares.
        """
        area: float = float(
            scaling_factor_area
            * gpd.GeoSeries([geometry], crs=4326).to_crs(epsg=epsg_area).area.values[0]
        )

        if self.verbose:
            print(f">>> Area of the input field: {area:.2f} hectares ...")

        return area

    def _save_dataset(
        self, ds: xr.Dataset, geometry: Polygon, start_date_year: str
    ) -> str:
        """
        Save the xarray Dataset to disk or S3 in Zarr or NetCDF format.

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

        # Patern example:
        # s3://agrilearn-xarray-datasets/minx_-46.208007_miny_-17.056277_maxx_-46.204046_maxy_-17.052840/2023/time-series/s2l8l9-raw.zarr/
        dataset_full_path: str = os.path.join(
            self.base_folder,
            f"{area_name}",
            f"{start_date_year}",
            "time-series",
            f"s2l8l9-raw.{self.save_format}",
        )

        if self.verbose:
            print(f">>> Saving dataset to {dataset_full_path} ...")

        if self.save_format == "zarr":
            ds.to_zarr(dataset_full_path, mode="w", consolidated=True)

        if self.save_format == "netcdf":
            ds.to_netcdf(dataset_full_path, mode="w")

        return dataset_full_path
