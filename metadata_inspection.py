import os
import xarray as xr
from config import get_nc_dir
import pprint
import argparse

def show_metadata(nc_path: str) -> dict:
    """
    Return a dictionary with metadata from a NetCDF file:
      - global attributes
      - dimensions
      - coordinates (with attrs)
      - data variables (dims, dtype, shape, attrs)
    """
    ds = xr.open_dataset(nc_path)

    metadata = {
        "attributes": dict(ds.attrs),
        "dimensions": dict(ds.dims),
        "coordinates": {},
        "data_vars": {},
    }

    for coord in ds.coords:
        da = ds.coords[coord]
        metadata["coordinates"][coord] = {
            "dims": da.dims,
            "length": da.size,
            "attrs": dict(da.attrs),
        }

    for var in ds.data_vars:
        da = ds[var]
        metadata["data_vars"][var] = {
            "dims": da.dims,
            "dtype": str(da.dtype),
            "shape": da.shape,
            "attrs": dict(da.attrs),
        }

    ds.close()
    return metadata

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inspect NetCDF metadata")
    parser.add_argument("nc_file", nargs="?", default=os.path.join(get_nc_dir(), "ERA5_daily.nc"),
                        help="Path to NetCDF file")
    args = parser.parse_args()

    nc_file = args.nc_file
    if not os.path.isabs(nc_file):
        nc_file = os.path.join(get_nc_dir(), nc_file)

    if not os.path.exists(nc_file):
        print(f"❌ File not found: {nc_file}")
    else:
        metadata = show_metadata(nc_file)
        pprint(metadata)  # Or save to JSON, print keys, etc.
