# Initial imports.
import ee
import json
import pathlib
from typing import Dict, Any


# Main parameters.
key_path: str = "/geosatlearn_app/gee_key_alternative_data.json"
folder_path: str = "projects/hblhug67j2mi/assets/bayer-api-tests"


# Ancillary function to initialize the Earth Engine library.
def gee_quick_init() -> None:
    gee_key_path: pathlib.Path = pathlib.Path(key_path)
    with open(gee_key_path) as gee_key_file:
        ee.Initialize(
            ee.ServiceAccountCredentials(
                json.load(gee_key_file)["client_email"], str(gee_key_path)
            ),
            opt_url="https://earthengine-highvolume.googleapis.com",
        )


# Initialize the Earth Engine library.
gee_quick_init()

# List all assets.
assets: Dict[str, Any] = ee.data.listAssets({"parent": folder_path})

# Delete all assets.
for asset in assets.get("assets", []):
    asset_id: str = asset["id"]
    print(f">>> Deleting asset: {asset_id}")
    ee.data.deleteAsset(asset_id)

print(">>> All assets deleted.")