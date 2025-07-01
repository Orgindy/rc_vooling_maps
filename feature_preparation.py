import os
import numpy as np
import pandas as pd
import xarray as xr
import logging
import rasterio
from pyproj import Transformer
from config import get_nc_dir, get_path
from humidity import compute_relative_humidity
from utils.sky_temperature import calculate_sky_temperature_improved
from utils.feature_utils import compute_band_ratios, filter_valid_columns
from pv_profiles import get_pv_cell_profiles
from pv_potential import calculate_pv_potential

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --------------------------------------
# 1. Physics-based PV potential function
# --------------------------------------

def load_netcdf_data(path: str, sample_fraction: float = 1.0) -> xr.Dataset:
    """Load a NetCDF file and optionally sample a fraction of its time steps."""
    with xr.open_dataset(path) as ds:
        data = ds.load()
    if 0 < sample_fraction < 1.0 and "time" in data:
        n = max(1, int(len(data.time) * sample_fraction))
        data = data.isel(time=slice(0, n))
    return data

def load_and_merge_datasets():
    logging.info("📥 Loading input datasets...")

    data_dir = get_nc_dir()
    rc_path = get_path("rc_potential_path")
    pv_path = get_path("pv_potential_path")
    spectral_csv_path = get_path("spectral_band_csv")

    era5 = xr.open_dataset(os.path.join(data_dir, "era5_2023_merged.nc"))
    rc_ds = xr.open_dataset(rc_path)
    pv_ds = xr.open_dataset(pv_path)

    dataset = xr.merge([era5, rc_ds, pv_ds], compat="override")
    logging.info("✅ Merged ERA5 + RC + PV datasets.")

    # Add spectral CSV bands
    spectral_df = pd.read_csv(spectral_csv_path)
    for band in ["Blue_band", "Green_band", "Red_band", "IR_band", "Total_band"]:
        if band in spectral_df.columns:
            band_array = spectral_df[band].values.reshape(len(dataset.latitude), len(dataset.longitude))
            dataset[band] = ("latitude", "longitude"), band_array
        else:
            logging.warning(f"⚠️ Missing spectral band in CSV: {band}")

    return dataset

def add_koppen_geiger_classification(df, kg_raster_path, lat_col="latitude", lon_col="longitude"):
    """Attach Köppen–Geiger climate classification code and label from a raster."""
    if not os.path.exists(kg_raster_path):
        logging.warning(f"⚠️ KG raster not found at {kg_raster_path}")
        df["KG_Code"] = np.nan
        df["KG_Label"] = np.nan
        return df

    coords = list(zip(df[lon_col], df[lat_col]))
    with rasterio.open(kg_raster_path) as src:
        if src.crs.to_string() != 'EPSG:4326':
            transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            coords = [transformer.transform(x, y) for x, y in coords]
        values = [val[0] if val.size > 0 else np.nan for val in src.sample(coords)]

    df["KG_Code"] = values

    # Optional: map code to climate label
    kg_lookup = {
        1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk',
        8: 'Csa', 9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
        14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
        20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa',
        26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 29: 'ET', 30: 'EF'
    }
    df["KG_Label"] = df["KG_Code"].map(kg_lookup)
    return df


def preprocess_dataset(dataset):
    logging.info("🔧 Preprocessing dataset...")

    # Compute RH if missing
    if "RH" not in dataset:
        if "t2m" in dataset and "d2m" in dataset:
            logging.info("🧮 Computing relative humidity...")
            dataset["RH"] = compute_relative_humidity(dataset["t2m"], dataset["d2m"], units="K")
        else:
            logging.warning("⚠️ Cannot compute RH — missing t2m or d2m")

    # Convert to DataFrame
    df = dataset.to_dataframe().reset_index()
    df = df.dropna(how="all")
    kg_path = get_path("kg_raster_path")  # add to config.yaml
    df = add_koppen_geiger_classification(df, kg_path)

    # Band ratios
    df, _ = compute_band_ratios(
        df,
        bands=["Blue_band", "Green_band", "Red_band", "IR_band"],
        total_col="Total_band"
    )

    logging.info(f"✅ Preprocessed dataset with {len(df)} rows.")
    return df

def save_feature_outputs(expanded_df, output_dir):
    """
    Save the full expanded dataset and a filtered ML-ready version.
    
    Parameters:
    - expanded_df (pd.DataFrame): Dataset with all tech-expanded features
    - output_dir (str): Directory to save output CSVs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define ML feature set
    ml_features = [
        # Climate
        "T_air", "RH", "Wind_Speed", "Cloud_Cover", "Dew_Point", "Albedo",
        # Irradiance
        "GHI", "RC_potential",
        # Spectral bands
        "Blue_band", "Green_band", "Red_band", "NIR_band", "IR_band", "Total_band",
        # Spectral ratios
        "Red_to_Total", "Blue_to_Total", "Green_to_Total", "IR_to_Total",
        # Technology-specific
        "tech_efficiency", "tech_temp_coeff", "spectral_factor"
    ]

    available_features = [f for f in ml_features if f in expanded_df.columns]
    missing_features = [f for f in ml_features if f not in expanded_df.columns]

    if missing_features:
        logging.warning(f"⚠️ Missing features in dataset: {missing_features}")
    else:
        logging.info("✅ All ML features found.")

    # Save full expanded dataset
    full_path = os.path.join(output_dir, "final_feature_data_multi_tech.csv")
    expanded_df.to_csv(full_path, index=False)
    logging.info(f"📦 Full expanded dataset saved to: {full_path}")

    # Save filtered ML dataset
    ml_df = expanded_df[available_features + ["PV_Potential", "Tech"]]
    ml_path = os.path.join(output_dir, "ml_feature_dataset_multi_tech.csv")
    ml_df.to_csv(ml_path, index=False)
    logging.info(f"✅ ML-ready dataset saved to: {ml_path}")
    # One-hot encode KG_Label if available
    if "KG_Label" in expanded_df.columns:
        logging.info("🏷️ Encoding Köppen–Geiger zones...")
        df_kg = expanded_df.copy()
        kg_dummies = pd.get_dummies(df_kg["KG_Label"], prefix="KG")
        df_kg = pd.concat([df_kg, kg_dummies], axis=1)
    
        # Extend ML features list
        kg_features = list(kg_dummies.columns)
        full_ml_features_kg = available_features + kg_features
    
        ml_kg_path = os.path.join(output_dir, "ml_feature_dataset_multi_tech_kg.csv")
        df_kg[full_ml_features_kg + ["PV_Potential", "Tech"]].to_csv(ml_kg_path, index=False)
        logging.info(f"✅ KG-encoded ML dataset saved to: {ml_kg_path}")
    else:
        logging.warning("⚠️ KG_Label column not found — skipping KG-encoded version.")


def validate_parameters(input_file, output_file, drop_invalid=True):
    """
    Validates climate, RC, and spectral parameters to ensure physical accuracy.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to the cleaned output file.
    - drop_invalid (bool): Whether to drop rows with invalid values (default True).
    
    Returns:
    - None
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    
    # Define valid ranges (based on physical limits)
    valid_ranges = {
        "GHI": (0, 1361),        # Max solar constant at TOA
        "T_air": (-90, 60),      # Realistic surface temperature range (°C)
        "RC_potential": (0, 300),  # Reasonable cooling range (W/m²)
        "Wind_Speed": (0, 150),  # Typical wind speeds (m/s)
        "Dew_Point": (-90, 60),  # Realistic dew point temperature (°C)
        "Blue_band": (0, 1500),  # Reasonable range for spectral bands (W/m²)
        "Green_band": (0, 1500),
        "Red_band": (0, 1500),
        "NIR_band": (0, 1500),
        "IR_band": (0, 1500),
        "Total_band": (0, 5000)  # Full spectral range (W/m²)
    }
    
    # Check each parameter
    invalid_rows = []
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            invalid_rows += df[(df[col] < min_val) | (df[col] > max_val)].index.tolist()
    
    # Drop or flag invalid rows
    if drop_invalid:
        df.drop(index=invalid_rows, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logging.info(f"✅ Dropped {len(invalid_rows)} rows with invalid values.")
    else:
        df["Invalid_Row"] = 0
        df.loc[invalid_rows, "Invalid_Row"] = 1
        logging.warning(f"⚠️ Flagged {len(invalid_rows)} rows as invalid.")
    
    # Save the cleaned file
    df.to_csv(output_file, index=False)
    logging.info(f"✅ Validated data saved to {output_file}")

def expand_dataset_for_all_technologies(df, pv_profiles):
    """
    Create one row per technology per location for PV potential estimation.

    Parameters:
    - df: original ERA5 + spectral + RC DataFrame
    - pv_profiles: dictionary of PV technology parameters

    Returns:
    - expanded_df: new DataFrame with one row per (location, time, tech)
    """
    tech_rows = []

    for tech, profile in pv_profiles.items():
        df_copy = df.copy()
        df_copy["Tech"] = tech
        df_copy["tech_efficiency"] = profile["efficiency"]
        df_copy["tech_temp_coeff"] = profile["temperature_coefficient"]
        red_opt = profile.get("red_ratio_optimum", 0.45)

        if "Red_band" in df_copy.columns and "Total_band" in df_copy.columns:
            red_ratio = df_copy["Red_band"] / df_copy["Total_band"]
            df_copy["spectral_factor"] = np.clip(red_ratio / red_opt, 0.7, 1.1)
        else:
            df_copy["spectral_factor"] = 1.0

        # Calculate PV potential using physics model
        df_copy["PV_Potential"] = calculate_pv_potential(
            df_copy["GHI"].values,
            df_copy["T_air"].values,
            df_copy["RC_potential"].values,
            df_copy["Red_band"].values if "Red_band" in df_copy else np.zeros(len(df_copy)),
            df_copy["Total_band"].values if "Total_band" in df_copy else np.ones(len(df_copy)),
            efficiency=profile["efficiency"],
            temp_coeff=profile["temperature_coefficient"],
            spectral_factor=df_copy["spectral_factor"].values
        )

        tech_rows.append(df_copy)

    expanded_df = pd.concat(tech_rows, ignore_index=True)
    logging.info(f"🔁 Expanded dataset to {len(expanded_df)} rows across {len(pv_profiles)} technologies.")
    return expanded_df

def main():
    """
    Entry point for full feature preparation pipeline.
    Uses:
    - ERA5 + RC + PV NetCDF (merged, multi-year)
    - Spectral band CSV
    - PV cell profiles
    - Outputs full and ML-ready multi-tech datasets
    """
    logging.info("🚀 Starting feature preparation from multi-year NetCDF input...")

    try:
        # Step 1: Load & merge
        dataset = load_and_merge_datasets()

        # Step 2: Preprocess & compute band ratios
        df_flat = preprocess_dataset(dataset)

        # Step 3: Expand for all PV technologies
        pv_profiles = get_pv_cell_profiles()
        df_expanded = expand_dataset_for_all_technologies(df_flat, pv_profiles)

        # Step 4: Save ML-ready output
        output_dir = get_path("results_path")
        save_feature_outputs(df_expanded, output_dir)

        logging.info(f"✅ Feature preparation completed. Rows: {len(df_expanded)}")

    except Exception as e:
        logging.error("❌ Feature preparation failed.")
        logging.exception(e)


if __name__ == "__main__":
    main()
