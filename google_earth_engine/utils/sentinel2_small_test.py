# Initial imports.
import json
import pathlib

import ee
import folium


def gee_quick_init():
    gee_key_path = pathlib.Path("/geosatlearn_app/gee_key_bucket_service_account.json")
    with open(gee_key_path) as gee_key_file:
        ee.Initialize(
            ee.ServiceAccountCredentials(
                json.load(gee_key_file)["client_email"], str(gee_key_path)
            ),
            opt_url="https://earthengine-highvolume.googleapis.com",
        )


def mask_s2_clouds(image):
    """
    Function to mask clouds using the Sentinel-2 QA band.
    :param image: ee.Image - Sentinel-2 image
    :return: ee.Image - Sentinel-2 image with cloud mask applied
    """
    qa = image.select("QA60")

    # Bits 10 and 11 represent clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags must be zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask).divide(10000)


# Initialize the Earth Engine library.
gee_quick_init()

# Define the input geometry (replace with your specific geometry).
geometry_input = ee.Geometry.Point([-46.66532912859609, -23.548325170254103]).buffer(
    1000
)

# Load Sentinel-2 TOA data and apply the cloud mask.
dataset = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterDate("2022-01-01", "2022-01-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_s2_clouds)
)

# Compute the median and clip to the input geometry.
dataset_cropped = dataset.median().clip(geometry_input)

# Count the number of pixels in the image.
pixel_count = (
    dataset_cropped.select("B4")
    .reduceRegion(
        reducer=ee.Reducer.count(), geometry=geometry_input, scale=10, maxPixels=1e13
    )
    .get("B4")
)

# Display the number of pixels in the image.
print(f">>> Number of pixels in the image: {pixel_count.getInfo()}")

# RGB visualization settings.
rgb_vis = {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]}

# Generate tile URL for visualization in folium.
map_id_dict = dataset_cropped.getMapId(rgb_vis)
tile_url = map_id_dict["tile_fetcher"].url_format

# Create a map with folium.
center_coords = [-23.548325170254103, -46.66532912859609]  # Central coordinates.
m = folium.Map(location=center_coords, zoom_start=15)

# Add Google Earth Engine layer to the map.
folium.TileLayer(
    tiles=tile_url, attr="Google Earth Engine", overlay=True, name="Sentinel-2 RGB"
).add_to(m)

# Add layer controls.
folium.LayerControl().add_to(m)

# Save the map to an HTML file.
file_name = "image_collection_small_test.html"
m.save(f"{file_name}")
print(f">>> Map saved as '{file_name}'. Open this file in a browser to view.")
