import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging
from config import get_nc_dir
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from sklearn.preprocessing import StandardScaler
from humidity import compute_relative_humidity
from utils.sky_temperature import calculate_sky_temperature_improved
import argparse
from visualize_rc_maps import run_visualization
from enhanced_thermal_model import solve_surface_temperature_scalar, get_material_properties
from constants import ATMOSPHERIC_CONSTANTS
from pv_profiles import RC_MATERIALS
from clustering import rc_only_clustering, plot_overlay_rc_pv_zones, generate_zone_descriptions
from config import get_path
from scipy.interpolate import griddata
from enhanced_thermal_model import solve_surface_temperature_scalar

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

##############################################################################
#                               CONFIGURATION
##############################################################################
STEFAN_BOLTZMANN = 5.67e-8      # W/m²/K^4
DEFAULT_RHO       = 0.2         # Default solar reflectivity
DEFAULT_EPS_COAT  = 0.95        # Assumed IR emissivity of the coating
CHUNK_SIZE        = 10000       # Number of rows per CSV chunk
KRIGING_SAMPLE_SIZE = 500       # Maximum number of points for training kriging
SIGMA = ATMOSPHERIC_CONSTANTS['sigma_sb']
SELECTED_MATERIAL = "Smart_Coating"  # or "Default_Coating"
material_config = RC_MATERIALS[SELECTED_MATERIAL]

# Optional: Load emissivity spectrum if given
EMISSIVITY_SPECTRUM = None  # default
if os.path.exists("my_emissivity.csv"):
    EMISSIVITY_SPECTRUM = pd.read_csv("my_emissivity.csv")
    logging.info(f"Loaded emissivity spectrum with {len(EMISSIVITY_SPECTRUM)} points.")
else:
    logging.info("No emissivity spectrum file found. Using default material settings.")


# Variogram model configuration for kriging
VARIOGRAM_MODEL = 'spherical'   # Options: 'spherical', 'exponential', 'gaussian', etc.

# Europe grid boundaries and resolution (min_lon, max_lon, step)
GRID_LON_RANGE = (-30, 40, 2.0)
# Europe grid boundaries and resolution (min_lat, max_lat, step)
GRID_LAT_RANGE = (35, 70, 2.0)

DATA_FOLDER = get_nc_dir()  # Load NetCDF path from config.yaml or NC_DATA_DIR

def planck_law(wavelength_um, T_kelvin):
    """Spectral blackbody exitance [W/m²/um]"""
    h = 6.62607015e-34  # J s
    c = 2.99792458e8    # m/s
    k = 1.380649e-23    # J/K

    wavelength_m = wavelength_um * 1e-6

    numerator = 2 * h * c ** 2
    exponent = (h * c) / (wavelength_m * k * T_kelvin)
    denominator = (wavelength_m ** 5) * (np.exp(exponent) - 1)

    B = numerator / denominator  # W/m²/m
    return B * 1e-6  # Convert to W/m²/um

def compute_eps_eff_from_spectrum(T_surface):
    wavelengths = EMISSIVITY_SPECTRUM['Wavelength_um'].values
    emissivities = EMISSIVITY_SPECTRUM['Emissivity'].values

    T_kelvin = T_surface + 273.15
    B_lambda = planck_law(wavelengths, T_kelvin)

    weighted = np.trapz(emissivities * B_lambda, wavelengths)
    unweighted = np.trapz(B_lambda, wavelengths)

    eps_eff = weighted / unweighted if unweighted > 0 else 0.9
    return eps_eff

def get_effective_albedo(nc_path: str, timestamp: pd.Timestamp, lat: float, lon: float) -> float:
    """
    Retrieve effective broadband albedo from a NetCDF file by interpolating the 'fal'
    (forecast albedo) variable to the given time and location.

    If the file cannot be read or interpolation fails, returns DEFAULT_RHO.
    """
    if not nc_path or not os.path.exists(nc_path):
        logging.info("NetCDF path not provided or not found; using default albedo.")
        return DEFAULT_RHO

    try:
        ds = xr.open_dataset(nc_path)

        if 'fal' not in ds:
            logging.warning("NetCDF file does not contain 'fal' (forecast albedo); using default albedo.")
            return DEFAULT_RHO

        ds = ds.sortby('time')

        # Interpolate to the specific timestamp, lat, lon
        albedo_interp = ds['fal'].interp(
            time=timestamp,
            latitude=lat,
            longitude=lon,
            method="nearest"
        )

        effective_albedo = float(albedo_interp.values)
        logging.info(f"Retrieved effective albedo {effective_albedo:.3f} from NetCDF.")
        return effective_albedo

    except Exception as e:
        logging.error(f"Error retrieving albedo from NetCDF: {e}")
        return DEFAULT_RHO

##############################################################################
#         QNET CALCULATION FUNCTIONS (Vectorized Implementation)
##############################################################################

def calculate_qnet_full(df):
    def _calc(row):
        GHI = row['msnwswrf']
        T_air = row['t2m'] - 273.15
        IR_down = row['msdwlwrf']
        wind = row['wind_speed']
        zenith = row['solar_zenith']

        props = get_material_properties(
            T_surface=T_air,
            GHI=GHI,
            solar_zenith=zenith,
            profile=material_config['switching_profile'],
            emissivity_profile=material_config['emissivity_profile'],
            alpha_profile=material_config['alpha_profile']
        )
        alpha = props['alpha']

        if EMISSIVITY_SPECTRUM is not None:
            epsilon = compute_eps_eff_from_spectrum(T_air)
            logging.debug(f"Using spectral ε_eff = {epsilon:.4f}")
        else:
            epsilon = props['emissivity']

        T_surf = solve_surface_temperature_scalar(
            GHI, T_air, IR_down, wind, alpha, epsilon,
            h_conv_base=5, h_conv_wind_coeff=4
        )

        T_sky = calculate_sky_temperature_improved(T_air, row['RH'], row['tcc'])
        Q_rad_out = epsilon * SIGMA * (T_surf + 273.15) ** 4
        Q_rad_in = epsilon * SIGMA * (T_sky + 273.15) ** 4
        Q_solar_abs = (1 - row['effective_albedo']) * GHI

        h_c = 5 + 4 * wind
        Q_conv = h_c * (T_surf - T_air)

        k = 0.5
        d = 0.002
        T_sub = T_air
        Q_cond = k * (T_surf - T_sub) / d

        return Q_rad_out - Q_rad_in - Q_solar_abs - Q_conv - Q_cond

    return df.apply(_calc, axis=1).to_numpy()

def calculate_qnet_vectorized(df: pd.DataFrame,
                              sigma: float = STEFAN_BOLTZMANN,
                              eps_coat: float = DEFAULT_EPS_COAT) -> pd.Series:
    """
    Vectorized QNET calculation using real ERA5 downward longwave flux (msdwlwrf)
    and realistic surface temperature from solve_surface_temperature_scalar().
    """

    # Albedo
    rho = df['effective_albedo'].fillna(DEFAULT_RHO) if 'effective_albedo' in df.columns else DEFAULT_RHO

    # Air temperature
    if 't2m' in df.columns:
        T_air = df['t2m'] - 273.15
    elif 'T_air' in df.columns:
        T_air = df['T_air']
    else:
        raise ValueError("Temperature column not found (expected 't2m' or 'T_air')")

    T_air_K = T_air + 273.15

    # Shortwave
    SW = df.get("msnwswrf", pd.Series(0.0, index=df.index))

    # Wind
    wind = df.get('wind_speed', pd.Series(2.0, index=df.index))

    # Use ERA5 IR down directly
    IR_down = df.get('msdwlwrf', pd.Series(400.0, index=df.index))  # fallback if missing

    # Radiative out
    Q_rad_out = eps_coat * sigma * (T_air_K ** 4)
    Q_rad_in  = eps_coat * IR_down
    Q_solar_abs = (1 - rho) * SW

    # === Real surface solve ===
    alpha = 0.2
    epsilon = eps_coat

    df['T_surface'] = df.apply(
        lambda row: solve_surface_temperature_scalar(
            row.get('msnwswrf', 800),
            row['T_air'] if 'T_air' in row else row['t2m'] - 273.15,
            row.get('msdwlwrf', 400),
            row.get('wind_speed', 2.0),
            alpha,
            epsilon
        ),
        axis=1
    )

    T_surf_K = df['T_surface'] + 273.15

    # Convective
    h_c = 5 + 4 * wind
    Q_conv = h_c * (T_surf_K - T_air_K)

    # Net
    qnet = Q_rad_out - Q_rad_in - Q_solar_abs - Q_conv

    return qnet


def add_effective_albedo_optimized(
    df: pd.DataFrame,
    nc_path: str,
    default_rho: float = DEFAULT_RHO
) -> pd.DataFrame:
    """
    Adds an 'effective_albedo' column to the DataFrame using 'fal' from a NetCDF file.
    Interpolates for each timestamp/lat/lon. Uses default if interpolation fails.
    """
    if not os.path.exists(nc_path):
        logging.error(f"NetCDF file not found: {nc_path}")
        df["effective_albedo"] = default_rho
        return df

    try:
        ds = xr.open_dataset(nc_path)
        if "fal" not in ds:
            logging.warning("'fal' not found in NetCDF; using default for all rows.")
            df["effective_albedo"] = default_rho
            return df

        ds = ds.sortby("time")

        try:
            times = pd.to_datetime(df["timestamp"])
            lats = xr.DataArray(df["lat"].astype(float).values, dims="points")
            lons = xr.DataArray(df["lon"].astype(float).values, dims="points")
            vals = ds["fal"].interp(
                time=("points", times),
                latitude=("points", lats),
                longitude=("points", lons),
                method="nearest",
            )
            df["effective_albedo"] = vals.values.astype(float)
        except Exception as e:
            logging.error(f"Vectorized albedo interpolation failed: {e}")
            df["effective_albedo"] = default_rho
        return df

    except Exception as e:
        logging.error(f"Failed to load or use NetCDF file: {e}")
        df["effective_albedo"] = default_rho
        return df

def save_qnet_to_netcdf(df: pd.DataFrame, output_path: str) -> None:
    """
    Save hourly QNET data to a NetCDF file using xarray.

    Parameters:
        df (pd.DataFrame): Must include ['time', 'lat', 'lon', 'QNET']
        output_path (str): File path to save the NetCDF
    """
    required = ['time', 'lat', 'lon', 'QNET']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Convert to xarray Dataset
    ds = df.set_index(['time', 'lat', 'lon']).to_xarray()

    # Add attributes
    ds['QNET'].attrs = {
        'units': 'W/m²',
        'long_name': 'Hourly Net Radiative Cooling',
        'description': 'Q_rad_out - Q_rad_in - Q_solar_abs'
    }
    ds.attrs['source'] = 'RC Cooling Model (ERA5-based)'
    ds.attrs['creator'] = 'YourName / ADAPTATION Project'
    
    # Save
    ds.to_netcdf(output_path)
    print(f"✅ Saved hourly QNET data to {output_path}")

def aggregate_qnet_rc_full(nc_path: str, output_dir: str, var_name: str = 'QNET'):
    """
    Aggregates hourly QNET NetCDF data to daily, monthly, and yearly scales,
    computing mean, sum (Wh, kWh), median, and max for each period.

    Saves one NetCDF file per aggregation level (daily, monthly, yearly).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds = xr.open_dataset(nc_path)
    if var_name not in ds:
        raise ValueError(f"{var_name} not found in {nc_path}")

    var = ds[var_name]  # e.g., QNET [W/m²]

    for freq, label in [('1D', 'daily'), ('1M', 'monthly'), ('1Y', 'yearly')]:
        group = var.resample(time=freq)

        mean = group.mean().rename(f"{var_name}_mean")
        sum_wh = (group.sum() * 1.0).rename(f"{var_name}_sum_Wh")     # 1h * W/m² = Wh/m²
        sum_kwh = (group.sum() / 1000.0).rename(f"{var_name}_sum_kWh")  # Convert Wh to kWh
        median = group.reduce(np.median).rename(f"{var_name}_median")
        vmax = group.max().rename(f"{var_name}_max")

        ds_out = xr.merge([mean, sum_wh, sum_kwh, median, vmax])
        out_file = os.path.join(output_dir, f"{label}_rc_metrics.nc")
        ds_out.to_netcdf(out_file)
        print(f"✅ Saved {label} RC metrics to {out_file}")

KRIGING_SAMPLE_SIZE = 500  # max samples for training
VARIOGRAM_MODEL = 'spherical'  # variogram model

def multi_year_kriging(
    netcdf_file,
    output_file,
    cluster_file=None,
    lat_col='latitude',
    lon_col='longitude',
    target_var='RC_total'
):
    """
    Perform kriging on yearly aggregated RC NetCDF data with:
    - train/test split,
    - training with random sample,
    - evaluation on test data,
    - prediction on full dataset.

    Parameters:
    - netcdf_file: str, path to aggregated NetCDF file
    - output_file: str, CSV path to save kriged data with predictions
    - cluster_file: optional path to CSV with cluster labels
    - lat_col, lon_col: coordinate column names
    - target_var: variable name in NetCDF to model (default 'RC_total')
    """

    # Load data
    ds = xr.open_dataset(netcdf_file)
    df = ds.to_dataframe().reset_index()

    if target_var not in df:
        raise ValueError(f"Target variable '{target_var}' not found in dataset")

    # Add clusters if available
    if cluster_file:
        cluster_df = pd.read_csv(cluster_file)
        df = pd.merge(df, cluster_df[[lat_col, lon_col, 'Cluster_ID']],
                      on=[lat_col, lon_col], how='left')

    def train_and_predict(df_subset):
        if len(df_subset) < 3:
            raise ValueError("Insufficient data points for Kriging")

        X = df_subset[[lon_col, lat_col]]
        y = df_subset[target_var]

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Sample training points for efficiency
        sample_size = min(len(X_train), KRIGING_SAMPLE_SIZE)
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.iloc[sample_indices])
        X_test_scaled = scaler.transform(X_test)

        y_train_sampled = y_train.iloc[sample_indices]

        # Train Ordinary Kriging
        OK = OrdinaryKriging(
            X_train_scaled[:, 0], X_train_scaled[:, 1], y_train_sampled.values,
            variogram_model=VARIOGRAM_MODEL,
            verbose=False, enable_plotting=False
        )

        # Predict on test set for evaluation
        y_pred, _ = OK.execute('points', X_test_scaled[:, 0], X_test_scaled[:, 1])
        rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
        logging.info(f"Test RMSE: {rmse:.4f} (samples: {sample_size})")

        # Predict on full set
        X_full_scaled = scaler.transform(X)
        z, _ = OK.execute('points', X_full_scaled[:, 0], X_full_scaled[:, 1])
        return z

    # Kriging predictions container
    predictions = []

    if 'Cluster_ID' in df.columns:
        for cluster_id in df['Cluster_ID'].dropna().unique():
            cluster_df = df[df['Cluster_ID'] == cluster_id]
            try:
                pred = train_and_predict(cluster_df)
                cluster_df = cluster_df.copy()
                cluster_df['RC_Kriged'] = pred
                predictions.append(cluster_df)
                logging.info(f"Kriging done for cluster {cluster_id}")
            except Exception as e:
                logging.error(f"Kriging failed for cluster {cluster_id}: {e}")
    else:
        # Global kriging if no clusters
        try:
            pred = train_and_predict(df)
            df['RC_Kriged'] = pred
            predictions.append(df)
            logging.info("Global kriging completed.")
        except Exception as e:
            logging.error(f"Global kriging failed: {e}")
            raise

    # Combine results and save
    final_df = pd.concat(predictions, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    logging.info(f"✅ Kriging result saved to {output_file}")


def estimate_sky_temperature_hybrid(df):
    """
    Estimate sky temperature (T_sky) in Celsius using ERA5 data with a hybrid emissivity model.

    Requires:
    - t2m: air temperature at 2m [K]
    - d2m: dew point at 2m [K]
    - tcc: total cloud cover [0–1]
    - msdwlwrf: mean surface downward longwave radiation flux [W/m²]

    Returns:
    - T_sky in Celsius
    """
    required_cols = ['t2m', 'd2m', 'tcc', 'msdwlwrf']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Temperatures already in Kelvin
    T_air_K = df['t2m']
    T_dew_K = df['d2m']

    # Relative Humidity
    RH = compute_relative_humidity(T_air_K, T_dew_K)  # imported from humidity.py
    logging.info("Using hybrid RH + cloud model for T_sky")

    # Hybrid sky emissivity model
    e_sky = 0.6 + 0.2 * df['tcc'] + 0.002 * RH
    e_sky = np.clip(e_sky, 0.7, 1.0)

    sigma = 5.670374419e-8  # Stefan–Boltzmann constant
    IR_down = df['msdwlwrf']  # W/m²

    T_sky_K = (IR_down / (e_sky * sigma)) ** 0.25
    return T_sky_K - 273.15  # Convert to Celsius

def calculate_and_save_day_night_rc_separate(nc_path: str, output_dir: str = "."):
    """
    From an hourly NetCDF file with RC and solar zenith data, calculate
    day and night radiative cooling (RC), aggregate to daily/monthly/yearly,
    and save 6 separate NetCDF files.
    
    Parameters:
    - nc_path (str): Path to the hourly input NetCDF file
    - output_dir (str): Directory where the 6 output files will be saved
    
    Requires:
    - Variables: 'RC' [W/m²], 'solar_zenith' [degrees], 'time', 'latitude', 'longitude'
    """
    
    ds = xr.open_dataset(nc_path)
    
    # Validate required inputs
    if 'RC' not in ds or 'solar_zenith' not in ds:
        raise ValueError("NetCDF must contain 'RC' and 'solar_zenith' variables.")

    rc = ds['RC']
    zenith = ds['solar_zenith']
    
    is_night = zenith > 90
    rc_day = rc.where(~is_night, 0.0)
    rc_night = rc.where(is_night, 0.0)

    # DAILY aggregation
    rc_day_daily = rc_day.resample(time="1D").sum()
    rc_night_daily = rc_night.resample(time="1D").sum()
    rc_day_daily.name = "RC_day_Wh"
    rc_night_daily.name = "RC_night_Wh"
    rc_day_daily.to_netcdf(f"{output_dir}/RC_day_daily.nc")
    rc_night_daily.to_netcdf(f"{output_dir}/RC_night_daily.nc")

    # MONTHLY aggregation
    rc_day_monthly = rc_day.resample(time="1M").sum() / 1000.0
    rc_night_monthly = rc_night.resample(time="1M").sum() / 1000.0
    rc_day_monthly.name = "RC_day_kWh"
    rc_night_monthly.name = "RC_night_kWh"
    rc_day_monthly.to_netcdf(f"{output_dir}/RC_day_monthly.nc")
    rc_night_monthly.to_netcdf(f"{output_dir}/RC_night_monthly.nc")

    # YEARLY aggregation
    rc_day_yearly = rc_day.resample(time="1Y").sum() / 1000.0
    rc_night_yearly = rc_night.resample(time="1Y").sum() / 1000.0
    rc_day_yearly.name = "RC_day_kWh"
    rc_night_yearly.name = "RC_night_kWh"
    rc_day_yearly.to_netcdf(f"{output_dir}/RC_day_yearly.nc")
    rc_night_yearly.to_netcdf(f"{output_dir}/RC_night_yearly.nc")

    logging.info("✅ Saved 6 RC cooling NetCDF files (day/night × daily/monthly/yearly)")


def main(netcdf_file: str,
         output_file: str,
         cluster_file: str = None,
         plot_dir: str = ".",
         mode: str = "vectorized"):
    try:
        logging.info(f"Loading ERA5 input data from: {netcdf_file}")
        ds = xr.open_dataset(netcdf_file)
        df = ds.to_dataframe().reset_index()
        logging.info(f"Loaded ERA5 grid data with {len(df)} rows.")

        # Merge cluster labels if provided
        if cluster_file and os.path.exists(cluster_file):
            cluster_df = pd.read_csv(cluster_file)
            df = pd.merge(df, cluster_df[['latitude', 'longitude', 'Cluster_ID']],
                          on=['latitude', 'longitude'], how='left')
            logging.info(f"Merged cluster labels from: {cluster_file}")

        # === Compute QNET ===
        logging.info(f"Calculating QNET using mode: {mode}")
        if mode == "full":
            df["QNET"] = calculate_qnet_full(df)
        else:
            df["QNET"] = calculate_qnet_vectorized(df)

        # Save QNET as NetCDF
        qnet_nc_path = output_file.replace(".csv", "_qnet.nc")
        save_qnet_to_netcdf(df, qnet_nc_path)
        logging.info(f"✅ Saved computed QNET to: {qnet_nc_path}")

        # === Kriging ===
        def train_and_evaluate(df_subset):
            X = df_subset[['longitude', 'latitude']]
            y = df_subset['QNET']  # Use new QNET here!
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            sample_size = min(len(X_train), 500)
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.iloc[idx])
            X_test_scaled = scaler.transform(X_test)
            y_train_sampled = y_train.iloc[idx]

            OK = OrdinaryKriging(
                X_train_scaled[:, 0], X_train_scaled[:, 1], y_train_sampled.values,
                variogram_model='spherical', verbose=False, enable_plotting=False)

            y_pred, _ = OK.execute('points', X_test_scaled[:, 0], X_test_scaled[:, 1])
            rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
            logging.info(f"Kriging test RMSE: {rmse:.4f} W/m²")

            X_full_scaled = scaler.transform(X)
            z, _ = OK.execute('points', X_full_scaled[:, 0], X_full_scaled[:, 1])
            return z

        predictions = []
        if 'Cluster_ID' in df.columns:
            clusters = df['Cluster_ID'].dropna().unique()
            for cluster_id in clusters:
                cluster_df = df[df['Cluster_ID'] == cluster_id]
                cluster_df = cluster_df.copy()
                try:
                    cluster_df['RC_Kriged'] = train_and_evaluate(cluster_df)
                    predictions.append(cluster_df)
                    logging.info(f"Kriging completed for cluster {cluster_id}")
                except Exception as e:
                    logging.error(f"Kriging failed for cluster {cluster_id}: {e}")
        else:
            df['RC_Kriged'] = train_and_evaluate(df)
            predictions.append(df)
            logging.info("Global kriging completed.")

        final_df = pd.concat(predictions, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        logging.info(f"Kriging results saved to {output_file}")

        # === Clustering ===

        try:
            model_path = get_path("rc_model_path")
            clustered_df, model, silhouette = rc_only_clustering(
                final_df,
                features=['RC_Kriged', 'T_air', 'RH', 'Wind_Speed'],  # ensure these exist
                model_path=model_path
            )
            logging.info(f"✅ RC clustering on kriged data complete. Silhouette score: {silhouette:.3f}")

            # Save clustered output
            clustered_path = get_path("rc_clustered_output")
            clustered_df.to_csv(clustered_path, index=False)
            logging.info(f"Clustered kriged RC data saved to {clustered_path}")

            # Visualization
            plot_overlay_rc_pv_zones(clustered_df)

            # Human-readable zone descriptions
            zone_descriptions = generate_zone_descriptions(clustered_df)
            zone_descriptions.to_csv(get_path("zone_description_csv"), index=False)

        except Exception as e:
            logging.error("❌ RC clustering on kriged data failed", exc_info=True)

        # === Plot ===
        logging.info("Plotting kriged RC potential map...")
        lon_vals = np.linspace(-30, 40, 71)
        lat_vals = np.linspace(35, 70, 36)
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

        points = final_df[['longitude', 'latitude']].values
        values = final_df['RC_Kriged'].values
        grid_z = griddata(points, values, (lon2d, lat2d), method='nearest')

        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        pcm = ax.pcolormesh(lon2d, lat2d, grid_z, cmap='YlOrRd', shading='auto', transform=ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_extent([-30, 40, 35, 70], crs=ccrs.PlateCarree())
        cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label("Kriged Annual RC Potential (W/m²)")
        ax.set_title("Kriged Annual Radiative Cooling Potential")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/rc_kriged_map.png", dpi=300)
        plt.show()

        logging.info("✅ RC cooling model execution completed successfully.")

    except Exception as e:
        logging.error("❌ An error occurred during execution.", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RC cooling kriging model")
    parser.add_argument("--netcdf-file", required=True, help="Path to yearly aggregated RC NetCDF file")
    parser.add_argument("--output-file", required=True, help="Path to save kriging output CSV")
    parser.add_argument("--cluster-file", default=None, help="Optional cluster CSV file path")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    parser.add_argument("--mode", default="vectorized", choices=["vectorized", "full"],
                        help="QNET calculation mode: 'vectorized' or 'full'")
    args = parser.parse_args()
    main(args.netcdf_file, args.output_file, args.cluster_file, args.plot_dir, args.mode)
