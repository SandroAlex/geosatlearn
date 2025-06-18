"""
This module provides tools for parsing Sentinel-2 Element84 STAC data into
xarray Datasets, with utilities for area calculation, catalog search, stacking,
and serialization. The main class, `Sentinel2Element84ToXarrayDataset`, enables
the conversion of geospatial field polygons into time-series xarray datasets
with relevant attributes, and saves them in Zarr format.
"""

# Initial imports.
import json
import os
from typing import List

import geopandas as gpd
import numpy as np
import pystac_client
import stackstac
import xarray as xr
from pystac.item import Item
from shapely import Polygon
from tqdm.auto import tqdm

from xdatastore.geometries.utils import create_area_name_from_geometry


class Sentinel2Element84ToXarrayDataset:
    """
    Parses Sentinel-2 Element84 STAC data into xarray Datasets.

    This class provides methods to process geospatial field polygons,
    retrieve Sentinel-2 imagery from a STAC API, stack the imagery into
    xarray DataArrays, and save the resulting datasets in Zarr format.

    Parameters
    ----------
    base_folder : str, optional
        Base folder or S3 path to save the output datasets.
    url : str, optional
        STAC API URL.
    collection : str, optional
        STAC collection name.
    assets : List[str], optional
        List of asset names to retrieve from the STAC items.
    verbose : bool, optional
        If True, prints progress and debug information.
    default_crs : int, optional
        EPSG code for the coordinate reference system.
    save_format : str, optional
        Format to save the datasets (default is "zarr").

    Examples
    --------
    >>> parser = Sentinel2Element84ToXarrayDataset()
    >>> output_gdf = parser.run(input_gdf)
    """

    def __init__(
        self,
        base_folder: str = "s3://agrilearn-xarray-datasets",
        url: str = "https://earth-search.aws.element84.com/v1",
        collection: str = "sentinel-2-l2a",
        assets: List[str] = [
            "blue",
            "green",
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "nir08",
            "swir16",
            "swir22",
            "scl",
        ],
        verbose: bool = True,
        default_crs: int = 4326,
        save_format: str = "zarr",
    ) -> None:
        """
        Initialize the parser with configuration parameters.

        Parameters
        ----------
        base_folder : str
            Base folder or S3 path to save the output datasets.
        url : str
            STAC API URL.
        collection : str
            STAC collection name.
        assets : List[str]
            List of asset names to retrieve from the STAC items.
        verbose : bool
            If True, prints progress and debug information.
        default_crs : int
            EPSG code for the coordinate reference system.
        save_format : str
            Format to save the datasets (default is "zarr").
        """

        # Main parameters.
        self.base_folder: str = base_folder
        self.url: str = url
        self.collection: str = collection
        self.assets: List[str] = assets
        self.verbose: bool = verbose
        self.default_crs: int = default_crs
        self.save_format: str = save_format

    def run(
        self,
        input_gdf: gpd.GeoDataFrame,
        start_date_column_name: str = "start_date",
        end_date_column_name: str = "end_date",
    ) -> gpd.GeoDataFrame:
        """
        Process each field in the input GeoDataFrame and save datasets.

        Parameters
        ----------
        input_gdf : geopandas.GeoDataFrame
            Input GeoDataFrame with field geometries and date columns.
        start_date_column_name : str, optional
            Name of the column with start dates.
        end_date_column_name : str, optional
            Name of the column with end dates.

        Returns
        -------
        geopandas.GeoDataFrame
            Output GeoDataFrame with an added column for dataset paths.
        """
        catalog: pystac_client.Client = self._initialize_stac_client()

        output_gdf: gpd.GeoDataFrame = input_gdf[
            ["geometry", start_date_column_name, end_date_column_name]
        ].copy(deep=True)

        output_gdf = output_gdf.to_crs(epsg=self.default_crs)

        output_fields_iterator = tqdm(
            range(len(output_gdf)), desc=f">>> Processing input field areas ..."
        )

        for intidx in output_fields_iterator:

            input_field_gdf: gpd.GeoDataFrame = output_gdf.iloc[[intidx]]
            area: float = self._calculate_area(input_field_gdf=input_field_gdf)
            geometry: Polygon = input_field_gdf.geometry.values[0]

            start_date_year: str = str(
                input_field_gdf[start_date_column_name].dt.year.values[0]
            )
            start_date: np.datetime64 = (
                input_field_gdf[start_date_column_name].values[0].astype(np.datetime64)
            )
            end_date: np.datetime64 = (
                input_field_gdf[end_date_column_name].values[0].astype(np.datetime64)
            )

            items: List[Item] = self._get_items_from_catalog(
                catalog=catalog,
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
            )

            if len(items) > 0:
                da: xr.DataArray = self._stack_all_available_items(
                    items=items, geometry=geometry
                )
                da = self._load_data_array_into_memory(da=da)
                da.attrs["provider_url"] = "https://earth-search.aws.element84.com/v1"
                da = self._parser_coordinates_and_attributes(da=da)
                ds: xr.Dataset = da.to_dataset(name="s2sr")
                ds.attrs["geometry_wkt"] = geometry.wkt
                ds.attrs["surface_area_ha"] = area

                save_path: str = self._save_dataset(
                    ds=ds, geometry=geometry, start_date_year=start_date_year
                )
                output_gdf.at[intidx, "dataset_full_path"] = save_path

                if self.verbose:
                    print(">>> Finished!\n")

        return output_gdf

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

    def _initialize_stac_client(self) -> pystac_client.Client:
        """
        Initialize the STAC client.

        Returns
        -------
        pystac_client.Client
            Initialized STAC client.
        """
        if self.verbose:
            print(f">>> Initializing STAC client with URL: {self.url}")

        catalog: pystac_client.Client = pystac_client.Client.open(self.url)
        return catalog

    def _get_items_from_catalog(
        self,
        catalog: pystac_client.Client,
        geometry: Polygon,
        start_date: np.datetime64,
        end_date: np.datetime64,
    ) -> List[Item]:
        """
        Search for STAC items in the catalog for the given geometry and dates.

        Parameters
        ----------
        catalog : pystac_client.Client
            STAC client instance.
        geometry : shapely.Polygon
            Geometry to search for.
        start_date : numpy.datetime64
            Start date for search.
        end_date : numpy.datetime64
            End date for search.

        Returns
        -------
        List[pystac.item.Item]
            List of STAC items found.
        """
        if self.verbose:
            print(f">>> Searching for items in the catalog ...")

        search = catalog.search(
            intersects=geometry,
            collections=[self.collection],
            datetime=f"{start_date}/{end_date}",
            limit=1000,
        )
        items: List[Item] = list(search.items())

        if self.verbose:
            print(f"* Found {len(items)} items ...")

        return items

    def _stack_all_available_items(
        self, items: List[Item], geometry: Polygon
    ) -> xr.DataArray:
        """
        Stack all available STAC items into a DataArray.

        Parameters
        ----------
        items : List[pystac.item.Item]
            List of STAC items to stack.
        geometry : shapely.Polygon
            Geometry bounding box for stacking.

        Returns
        -------
        xarray.DataArray
            Stacked DataArray of imagery.
        """
        if self.verbose:
            print(f"* Stacking all available items ...")

        minx, miny, maxx, maxy = geometry.bounds

        da: xr.DataArray = stackstac.stack(
            items=items,
            assets=self.assets,
            bounds=(minx, miny, maxx, maxy),
            epsg=self.default_crs,
            dtype=np.float64,
            fill_value=np.nan,
            rescale=True,
        )

        if self.verbose:
            print(f"* Stacked DataArray with shape {da.shape} ...")

        return da

    def _load_data_array_into_memory(self, da: xr.DataArray) -> xr.DataArray:
        """
        Load a DataArray into memory.

        Parameters
        ----------
        da : xarray.DataArray
            DataArray to load.

        Returns
        -------
        xarray.DataArray
            Loaded DataArray.
        """
        if self.verbose:
            print(f">>> Loading DataArray into memory ...")

        da = da.load()
        return da

    def _parser_coordinates_and_attributes(self, da: xr.DataArray) -> xr.DataArray:
        """
        Remove non-serializable coordinates and attributes from a DataArray.

        Parameters
        ----------
        da : xarray.DataArray
            DataArray to clean.

        Returns
        -------
        xarray.DataArray
            Cleaned DataArray with only serializable attributes and coordinates.
        """
        coords_to_be_deleted: List[str] = []
        for coord in da.coords:
            if da[coord].dtype == "object":
                coords_to_be_deleted.append(coord)

        for coord in coords_to_be_deleted:
            da = da.drop_vars(coord)

        attrs_to_be_deleted: List[str] = []
        for attr in da.attrs:
            try:
                json.dumps(da.attrs[attr])
            except Exception:
                attrs_to_be_deleted.append(attr)

        for attr in attrs_to_be_deleted:
            del da.attrs[attr]

        return da

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

        # Patern example:
        # s3://agrilearn-xarray-datasets/minx_-46.208007_miny_-17.056277_maxx_-46.204046_maxy_-17.052840/2023/images/s2sr-raw.zarr/
        dataset_full_path: str = os.path.join(
            self.base_folder,
            f"{area_name}",
            f"{start_date_year}",
            "images",
            f"s2sr-raw.{self.save_format}",
        )

        if self.verbose:
            print(f">>> Saving dataset to {dataset_full_path} ...")

        if self.save_format == "zarr":
            ds.to_zarr(dataset_full_path, mode="w", consolidated=True)

        if self.save_format == "netcdf":
            ds.to_netcdf(dataset_full_path, mode="w")

        return dataset_full_path
