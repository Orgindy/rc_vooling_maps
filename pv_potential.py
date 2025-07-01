import numpy as np
import logging
import warnings
from typing import Union, Tuple
import xarray as xr
from pathlib import Path
from config import get_path
from constants import PHYSICAL_LIMITS
from pv_profiles import PV_CONSTANTS
from pykrige.ok import OrdinaryKriging
import pvlib
from enhanced_thermal_model import compute_temperature_series
import pandas as pd

logger = logging.getLogger(__name__)

def load_pv_technology_profiles(csv_path: Union[str, Path]) -> dict:
    """Load PV technology profiles from CSV file."""

    df = pd.read_csv(csv_path)
    profiles = {}
    for _, row in df.iterrows():
        tech = row.get("Technology")
        if not tech:
            continue
        profiles[tech] = {
            "temperature_coefficient": row.get("Temperature_Coefficient", -0.004),
            "stc_efficiency": row.get("STC_Efficiency", 0.2),
            "reference_red_fraction": row.get(
                "Reference_Red_Fraction",
                PV_CONSTANTS.get("Reference_Red_Fraction", 0.42),
            ),
        }

    return profiles

# Load profiles once using configurable path
PV_TECH_CSV = get_path(
    "pv_tech_profiles_path",
    Path(__file__).with_name("pv_technology_profiles_enhanced.csv")
)
pv_tech_profiles = load_pv_technology_profiles(str(PV_TECH_CSV))

# Extract technology names and parameter arrays for use in calculations
tech_names = list(pv_tech_profiles.keys())
temp_coeffs = np.array([pv_tech_profiles[t]['temperature_coefficient'] for t in tech_names])
ref_red_fractions = np.array([pv_tech_profiles[t]['reference_red_fraction'] for t in tech_names])

# Material config for thermal model (adjustable)
material_config = {
    "alpha_solar": 0.9,
    "epsilon_IR": 0.92,
    "thickness_m": 0.003,
    "density": 2500,
    "cp": 900,
    "h_conv_base": 5,
    "h_conv_wind_coeff": 4,
    "use_dynamic_emissivity": False,
}

# PVLib module temperature parameters (adjustable)
module_params = {
    'a': -3.56,    # Wind speed coefficient [°C m/s]
    'b': 0.94,     # Irradiance coefficient [°C / kW/m²]
    'deltaT': 3,   # Temperature offset [°C]
    'altitude': 0  # Altitude above sea level [m]
}

def load_era5_data() -> xr.Dataset:
    era5_path = get_path("DATA_FOLDER")
    if era5_path is None:
        raise FileNotFoundError("ERA5 data path not found in config.")
    ds = xr.open_dataset(era5_path)
    return ds

def calculate_pv_potential_multi_tech(
    GHI: np.ndarray,
    T_cell: np.ndarray,
    RC_potential: np.ndarray,
    Red_band: np.ndarray,
    Total_band: np.ndarray,
    temp_coeffs: np.ndarray,
    ref_red_fractions: np.ndarray,
    PR_ref: float = 0.9,
    PR_bounds: tuple = (0.0, 1.0),
) -> np.ndarray:
    """
    Calculate PV potential for multiple technologies simultaneously.

    Parameters:
        GHI: np.ndarray, shape (..., tech)
        T_cell: np.ndarray, shape (..., tech)
        RC_potential: np.ndarray, shape (..., tech)
        Red_band: np.ndarray, shape (..., tech)
        Total_band: np.ndarray, shape (..., tech)
        temp_coeffs: np.ndarray, shape (tech,)
        ref_red_fractions: np.ndarray, shape (tech,)
        PR_ref: float, reference performance ratio
        PR_bounds: tuple, (min_PR, max_PR) bounds to clip results

    Returns:
        PV potential array with shape (..., tech)
    """

    # Validate temperature coefficients shape
    if temp_coeffs.shape[0] != GHI.shape[-1]:
        raise ValueError("Temperature coefficients array size must match the technology dimension size.")

    # Broadcast temp_coeffs and ref_red_fractions to input shape
    expand_dims = tuple(range(GHI.ndim - 1))
    temp_coeffs_exp = np.expand_dims(temp_coeffs, axis=expand_dims)
    ref_red_exp = np.expand_dims(ref_red_fractions, axis=expand_dims)

    # Calculate temperature loss term
    Temp_Loss = temp_coeffs_exp * (T_cell - 25)

    # Calculate radiative cooling gain term
    RC_Gain = 0.01 * (RC_potential / 50)

    # Calculate spectral adjustment safely
    with np.errstate(divide='ignore', invalid='ignore'):
        Actual_Red_Fraction = np.divide(
            Red_band,
            Total_band,
            out=np.full_like(Red_band, ref_red_exp),
            where=(Total_band > 0) & ~np.isnan(Total_band),
        )
    Spectral_Adjust = Actual_Red_Fraction - ref_red_exp

    # Calculate corrected performance ratio
    PR_corrected = PR_ref + Temp_Loss + RC_Gain + Spectral_Adjust

    # Clip PR to physical bounds
    PR_corrected = np.clip(PR_corrected, PR_bounds[0], PR_bounds[1])

    # Calculate PV potential
    PV_Potential = GHI * PR_corrected

    # Replace NaNs with zero for safety
    PV_Potential = np.nan_to_num(PV_Potential, nan=0.0)

    return PV_Potential

def validate_pv_inputs(
    GHI: np.ndarray,
    T_air: np.ndarray,
    RC_potential: np.ndarray,
    Red_band: np.ndarray,
    Total_band: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs for PV potential calculation."""
    inputs = {
        'GHI': (GHI, *PHYSICAL_LIMITS['GHI']),
        'T_air': (T_air, *PHYSICAL_LIMITS['T_air']),
        'RC_potential': (RC_potential, *PHYSICAL_LIMITS['RC_potential']),
        'Red_band': (Red_band, *PHYSICAL_LIMITS['Red_band']),
        'Total_band': (Total_band, *PHYSICAL_LIMITS['Total_band']),
    }

    validated = {}
    for name, (value, min_val, max_val) in inputs.items():
        arr = np.asarray(value, dtype=float)
        if min_val is not None and np.any(arr < min_val):
            warnings.warn(
                f"{name} values below {min_val} clamped", stacklevel=2
            )
            logger.warning("%s values below %s clamped", name, min_val)
            arr = np.clip(arr, min_val, None)
        if max_val is not None and np.any(arr > max_val):
            warnings.warn(
                f"{name} values above {max_val} clamped", stacklevel=2
            )
            logger.warning("%s values above %s clamped", name, max_val)
            arr = np.clip(arr, None, max_val)
        validated[name] = arr

    return (
        validated['GHI'],
        validated['T_air'],
        validated['RC_potential'],
        validated['Red_band'],
        validated['Total_band'],
    )


logger = logging.getLogger(__name__)

def validate_temperature_coefficient(temp_coeff: float) -> float:
    """Validate and clamp temperature coefficient within realistic bounds."""
    try:
        coeff = float(temp_coeff)
    except Exception:
        warnings.warn(
            "Temperature coefficient must be numeric, using default -0.004",
            stacklevel=2,
        )
        logger.warning("Invalid temperature coefficient '%s', using default", temp_coeff)
        return -0.004

    if coeff < -0.01:
        warnings.warn("Temperature coefficient below -0.01, clamping", stacklevel=2)
        logger.warning("Temperature coefficient %s below -0.01, clamping", coeff)
        coeff = -0.01
    elif coeff > 0:
        warnings.warn("Temperature coefficient above 0, clamping", stacklevel=2)
        logger.warning("Temperature coefficient %s above 0, clamping", coeff)
        coeff = 0.0

    return coeff

def calculate_pv_potential(
    GHI: Union[float, np.ndarray],
    T_cell: Union[float, np.ndarray] = None,
    RC_potential: Union[float, np.ndarray] = 0,
    Red_band: Union[float, np.ndarray] = 0,
    Total_band: Union[float, np.ndarray] = 1,
    temp_coeff: float = -0.004,
    PR_ref: float = 0.9,
    PR_bounds: tuple = (0.0, 1.0),
    T_air: Union[float, np.ndarray, None] = None,
) -> np.ndarray:
    """
    Calculate PV potential given physics-based module temperature and spectral features.

    Parameters:
        GHI: Global horizontal irradiance [W/m²]
        T_cell: Module temperature [°C]. If ``None``, ``T_air`` is used for
            backward compatibility.
        RC_potential: Radiative cooling potential [W/m²]
        Red_band: Spectral irradiance in red band [W/m²]
        Total_band: Total spectral irradiance [W/m²]
        temp_coeff: Temperature coefficient for efficiency loss [per °C]
        PR_ref: Reference performance ratio (unitless)
        PR_bounds: Tuple (min_PR, max_PR) clipping bounds for performance ratio

    Returns:
        PV potential array [W/m²]
    """

    if T_cell is None:
        if T_air is None:
            raise TypeError("Either T_cell or T_air must be provided")
        warnings.warn("T_air argument is deprecated; use T_cell", UserWarning)
        T_cell = T_air

    # Validate inputs
    temp_coeff = validate_temperature_coefficient(temp_coeff)
    GHI, T_cell, RC_potential, Red_band, Total_band = validate_pv_inputs(
        GHI, T_cell, RC_potential, Red_band, Total_band
    )

    # Calculate temperature loss term (linear efficiency loss)
    Temp_Loss = temp_coeff * (T_cell - 25)

    # Simplified radiative cooling gain term (empirical)
    RC_Gain = 0.01 * (RC_potential / 50)

    # Calculate spectral adjustment safely
    with np.errstate(divide='ignore', invalid='ignore'):
        Actual_Red_Fraction = np.divide(
            Red_band,
            Total_band,
            out=np.full_like(Red_band, 0.35),  # Example default red fraction; adjust as needed
            where=(Total_band > 0) & ~np.isnan(Total_band),
        )
    Spectral_Adjust = Actual_Red_Fraction - 0.35  # Adjust relative to reference

    # Calculate corrected performance ratio (PR)
    PR_corrected = PR_ref + Temp_Loss + RC_Gain + Spectral_Adjust

    # Clamp PR to physical bounds
    min_pr, max_pr = PR_bounds
    PR_corrected = np.clip(PR_corrected, min_pr, max_pr)

    # Calculate PV potential
    PV_Potential = GHI * PR_corrected

    # Replace NaNs with zeros for safety
    PV_Potential = np.nan_to_num(PV_Potential, nan=0.0)

    return PV_Potential

def kriging_on_pv_potential_multi_tech(pv_potential_multi, times, lats, lons, tech_names):
    """
    Apply kriging smoothing to each technology layer separately.

    Parameters:
        pv_potential_multi: np.ndarray, shape (time, lat, lon, tech)
        times: np.ndarray
        lats: np.ndarray
        lons: np.ndarray
        tech_names: list of str

    Returns:
        np.ndarray, same shape (time, lat, lon, tech)
    """
    smoothed_all_techs = []

    for i, tech in enumerate(tech_names):
        # Wrap slice as xarray DataArray
        da = xr.DataArray(
            pv_potential_multi[..., i],
            coords={'time': times, 'latitude': lats, 'longitude': lons},
            dims=['time', 'latitude', 'longitude']
        )
        smoothed = kriging_on_pv_potential(da)
        smoothed_all_techs.append(smoothed.values)

    # Stack along tech dimension
    smoothed_all_techs = np.stack(smoothed_all_techs, axis=-1)
    return smoothed_all_techs

def kriging_smooth(latitudes, longitudes, values, grid_lat, grid_lon):
    # Create kriging object
    OK = OrdinaryKriging(
        longitudes, latitudes, values,
        variogram_model='exponential',
        verbose=False,
        enable_plotting=False,
    )
    # Predict on grid
    z, ss = OK.execute('grid', grid_lon, grid_lat)
    return z

def kriging_on_pv_potential(pv_potential_da):
    smoothed_data = []

    lats = pv_potential_da['latitude'].values
    lons = pv_potential_da['longitude'].values

    grid_lat = np.linspace(lats.min(), lats.max(), len(lats))
    grid_lon = np.linspace(lons.min(), lons.max(), len(lons))

    for t in pv_potential_da['time'].values:
        values = pv_potential_da.sel(time=t).values.flatten()
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()

        # Remove nan
        mask = ~np.isnan(values)
        z = kriging_smooth(lat_flat[mask], lon_flat[mask], values[mask], grid_lat, grid_lon)
        smoothed_data.append(z)

    smoothed_array = np.array(smoothed_data)
    # Create xarray DataArray with smoothed data and original coords/time
    smoothed_da = xr.DataArray(
        smoothed_array,
        coords=[pv_potential_da['time'], grid_lat, grid_lon],
        dims=['time', 'latitude', 'longitude'],
        name='PV_potential_smoothed'
    )
    return smoothed_da

def compute_and_aggregate_pv(use_pvlib: bool = False):
    """
    Full pipeline:
    - Load ERA5
    - Compute multi-tech PV potential
    - Use enhanced thermal model OR pvlib for module temperature
    - Aggregate & save NetCDFs
    """

    ds = xr.open_dataset(get_path("DATA_FOLDER"))

    ghi = ds['GHI'].values
    tair = ds['T2m'].values  # Kelvin
    ir_down = ds['IR_down'].values
    wind = ds['Wind'].values
    rc_potential = ds['RC'].values
    red_band = ds['RedBand'].values
    total_band = ds['TotalBand'].values

    tech_count = len(temp_coeffs)

    # Expand each input to tech dimension
    def expand_tech_dim(arr):
        return np.repeat(arr[..., np.newaxis], tech_count, axis=-1)

    rc_potential_multi = expand_tech_dim(rc_potential)
    red_band_multi = expand_tech_dim(red_band)
    total_band_multi = expand_tech_dim(total_band)

    # Get grid coordinates
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    times = ds['time'].values
    
    print("🟢 Using enhanced thermal model for module temperature.")
    ghi_multi = expand_tech_dim(ghi)

    module_temp = compute_temperature_series(
        ghi, tair, ir_down, wind, material_config
    )
    T_cell_celsius_multi = expand_tech_dim(module_temp - 273.15)

    # Calculate PV potential for all technologies
    pv_potential_multi = calculate_pv_potential_multi_tech(
        GHI=ghi_multi,
        T_cell=T_cell_celsius_multi,
        RC_potential=rc_potential_multi,
        Red_band=red_band_multi,
        Total_band=total_band_multi,
        temp_coeffs=temp_coeffs,
        ref_red_fractions=ref_red_fractions,
    )

    # Save with daily/monthly/yearly aggregation
    aggregate_and_save_multi_tech(
        pv_potential_multi,
        times,
        lats,
        lons,
        tech_names,
        output_dir=get_path("results_path")
    )

    print(f"✅ PV potential calculated and saved. pvlib={use_pvlib}")

def aggregate_and_save_multi_tech(pv_potential_multi, times, lats, lons, tech_names, output_dir):
    # Create xarray DataArray for multi-tech PV potential
    da = xr.DataArray(
        pv_potential_multi,
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
            'technology': tech_names
        },
        dims=['time', 'latitude', 'longitude', 'technology']
    )

    # Daily aggregation
    daily = da.resample(time='1D').sum()
    daily.to_netcdf(f"{output_dir}/pv_potential_daily_multi_tech.nc")

    # Monthly aggregation
    monthly = da.resample(time='1M').sum()
    monthly.to_netcdf(f"{output_dir}/pv_potential_monthly_multi_tech.nc")

    # Yearly aggregation
    yearly = da.resample(time='1Y').sum()
    yearly.to_netcdf(f"{output_dir}/pv_potential_yearly_multi_tech.nc")

    print(f"Aggregated PV potential saved to {output_dir}")
