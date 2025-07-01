#!/usr/bin/env python3
"""
verify_netcdf.py

Enhanced NetCDF verification utility with CSV summary output.

Checks:
- Required coordinates: latitude, longitude, time
- Required variables: user-defined
- Dimensions, time range, spatial resolution, metadata

Usage:
    python verify_netcdf.py --directory ./data/netcdf_inputs --variables GHI T_air RC_potential --output results/netcdf_verification.csv
"""

import os
import argparse
import logging
from pathlib import Path
import csv

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_spatial_resolution(coord):
    values = coord.values
    if len(values) < 2:
        return None
    diffs = np.diff(values)
    step = np.median(np.abs(diffs))
    return step


def verify_netcdf_file(path, required_vars):
    info = {
        "file": os.path.basename(path),
        "valid": False,
        "error": None,
        "dimensions": "",
        "variables": "",
        "time_range": "",
        "lat_res": None,
        "lon_res": None
    }

    try:
        ds = xr.open_dataset(path)

        required_coords = {"latitude", "longitude", "time"}
        if not required_coords.issubset(ds.coords):
            info["error"] = f"Missing coordinates: {required_coords - set(ds.coords)}"
            return info

        present_vars = set(ds.data_vars)
        missing = [var for var in required_vars if var not in present_vars]
        if missing:
            info["error"] = f"Missing variables: {missing}"
            return info

        dims = dict(ds.dims)
        info["dimensions"] = "; ".join(f"{k}:{v}" for k, v in dims.items())
        info["variables"] = ", ".join(ds.data_vars)

        if "time" in ds.coords:
            time_vals = ds["time"].values
            if len(time_vals) > 0:
                info["time_range"] = f"{str(time_vals[0])} to {str(time_vals[-1])}"

        info["lat_res"] = get_spatial_resolution(ds["latitude"])
        info["lon_res"] = get_spatial_resolution(ds["longitude"])

        info["valid"] = True

    except Exception as e:
        info["error"] = str(e)

    finally:
        ds.close()

    return info


def find_netcdf_files(directory):
    return sorted(Path(directory).glob("*.nc"))


def main(directory, required_vars, output_csv):
    files = find_netcdf_files(directory)

    if not files:
        logging.warning(f"No NetCDF files found in {directory}")
        return

    logging.info(f"🔍 Found {len(files)} NetCDF file(s) in {directory}\n")

    results = []

    for fpath in files:
        result = verify_netcdf_file(fpath, required_vars)

        print(f"File: {result['file']}")
        if result["valid"]:
            print(f"  ✅ Valid")
            print(f"  Dimensions : {result['dimensions']}")
            print(f"  Variables  : {result['variables']}")
            if result["time_range"]:
                print(f"  Time range : {result['time_range']}")
            if result["lat_res"] is not None and result["lon_res"] is not None:
                print(f"  Spatial resolution : {result['lat_res']}° lat x {result['lon_res']}° lon")
        else:
            print(f"  ❌ Invalid: {result['error']}")
        print()

        results.append({
            "file": result["file"],
            "valid": result["valid"],
            "dimensions": result["dimensions"],
            "variables": result["variables"],
            "time_range": result["time_range"],
            "lat_res_deg": result["lat_res"],
            "lon_res_deg": result["lon_res"],
            "error": result["error"] if result["error"] else ""
        })

    if output_csv:
        output_dir = os.path.dirname(output_csv)
        os.makedirs(output_dir, exist_ok=True)

        keys = ["file", "valid", "dimensions", "variables", "time_range", "lat_res_deg", "lon_res_deg", "error"]
        with open(output_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        logging.info(f"✅ Saved summary CSV: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify NetCDF files and save summary CSV.")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to folder containing NetCDF files."
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        required=True,
        help="List of required variable names (e.g., GHI T_air RC_potential)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output summary CSV (e.g., results/netcdf_verification.csv)"
    )

    args = parser.parse_args()

    main(args.directory, args.variables, args.output)
