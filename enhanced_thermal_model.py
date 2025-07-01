# -*- coding: utf-8 -*-
"""
Enhanced Thermal Model — Updated
Vectorized-ready, config-connected version.
"""
import numpy as np
from scipy.optimize import root_scalar
from dynamic_materials import get_material_properties
from pv_profiles import RC_MATERIALS
import xarray as xr
import numpy as np
from pv_profiles import PV_CONSTANTS

# Optional: config connection
try:
    from config import get_config
except ImportError:
    get_config = None  # fallback if you want to run standalone

# Stefan–Boltzmann constant
SIGMA = 5.670374419e-8  # W/m²·K⁴
SELECTED_MATERIAL = "Smart_Coating"
material_config = RC_MATERIALS[SELECTED_MATERIAL]

def compute_temperature_series(
    ghi_array,
    tair_array,
    ir_down_array,
    wind_array,
    zenith_array=None
):
    """
    Compute surface temperature time series for your RC layer.
    Uses RC_MATERIALS config.
    Supports dynamic switching if profiles are present.
    """
    n = len(ghi_array)
    T_surface_series = np.zeros(n)

    # Decide if this coating has dynamic switching
    use_dynamic = all(
        key in material_config for key in ["switching_profile", "emissivity_profile", "alpha_profile"]
    ) and zenith_array is not None

    for i in range(n):
        GHI = ghi_array[i]
        T_air = tair_array[i]
        IR_down = ir_down_array[i]
        wind_speed = wind_array[i]

        if use_dynamic:
            solar_zenith = zenith_array[i]
            props = get_material_properties(
                T_surface=T_air,  # initial guess for switching
                GHI=GHI,
                solar_zenith=solar_zenith,
                profile=material_config["switching_profile"],
                emissivity_profile=material_config["emissivity_profile"],
                alpha_profile=material_config["alpha_profile"]
            )

            dynamic_config = material_config.copy()
            dynamic_config["alpha_solar"] = props["alpha"]
            dynamic_config["epsilon_IR"] = props["emissivity"]

            config_to_use = dynamic_config
            print(f"[{i}] Dynamic → α: {props['alpha']:.3f}, ε: {props['emissivity']:.3f}, State: {props['state']}")

        else:
            config_to_use = material_config
            print(f"[{i}] Static → α: {config_to_use['alpha_solar']:.3f}, ε: {config_to_use['epsilon_IR']:.3f}")

        try:
            T_surf = solve_surface_temperature_scalar(
                GHI,
                T_air,
                IR_down,
                wind_speed,
                alpha=config_to_use["alpha_solar"],
                epsilon=config_to_use["epsilon_IR"],
                h_conv_base=config_to_use["h_conv_base"],
                h_conv_wind_coeff=config_to_use["h_conv_wind_coeff"]
            )
        except RuntimeError as e:
            print(f"[{i}] Solver failed: {e}")
            T_surf = np.nan

        T_surface_series[i] = T_surf

    return T_surface_series

def solve_surface_temperature_scalar(
    GHI,
    T_air,
    IR_down,
    wind_speed,
    alpha,
    epsilon,
    h_conv_base,
    h_conv_wind_coeff
):
    """
    Scalar version for one point in time.
    """
    T_sky = (IR_down / (epsilon * SIGMA)) ** 0.25

    def energy_balance(T_surface):
        Q_solar = alpha * GHI
        h = h_conv_base + h_conv_wind_coeff * wind_speed
        Q_conv = h * (T_surface - T_air)
        Q_emit = epsilon * SIGMA * T_surface**4
        Q_absorb = epsilon * SIGMA * T_sky**4
        Q_rad = Q_emit - Q_absorb
        return Q_solar - Q_conv - Q_rad

    try:
        result = root_scalar(
            energy_balance,
            bracket=[T_air - 30, T_air + 60],
            method="brentq",
            maxiter=100
        )
        return result.root if result.converged else np.nan
    except Exception:
        return np.nan


def compute_temperature_vectorized(
    GHI, T_air, IR_down, wind_speed, config
):
    """
    Vectorized version for static RC coating.
    Uses xarray.apply_ufunc to map scalar solver.
    """

    alpha = config["alpha_solar"]
    epsilon = config["epsilon_IR"]
    h_conv_base = config["h_conv_base"]
    h_conv_wind_coeff = config["h_conv_wind_coeff"]

    T_surface = xr.apply_ufunc(
        solve_surface_temperature_scalar,
        GHI,
        T_air,
        IR_down,
        wind_speed,
        input_core_dims=[[], [], [], []],
        kwargs={
            "alpha": alpha,
            "epsilon": epsilon,
            "h_conv_base": h_conv_base,
            "h_conv_wind_coeff": h_conv_wind_coeff,
        },
        vectorize=True,  # <- important!
        dask="parallelized",
        output_dtypes=[float],
    )

    return T_surface

def estimate_pv_cell_temperature(
    GHI,
    T_air,
    wind_speed,
    tech_name="Silicon",
    model="NOCT"
):
    """
    Estimate PV cell temperature for a given technology using NOCT or Sandia model.

    Parameters
    ----------
    GHI : float or np.ndarray
        Global horizontal irradiance [W/m²].
    T_air : float or np.ndarray
        Ambient air temperature [°C].
    wind_speed : float or np.ndarray
        Wind speed at module surface [m/s].
    tech_name : str
        PV technology key from PV_CONSTANTS.
    model : str
        Either 'NOCT' or 'Sandia'.

    Returns
    -------
    T_cell : float or np.ndarray
        Estimated PV cell temperature [°C].
    """

    # Get NOCT for this PV type (fallback to 45 if missing)
    noct = PV_CONSTANTS.get("NOCT", 45)

    if model == "NOCT":
        # Standard empirical NOCT model
        G_ref = 800  # Reference irradiance [W/m²]
        T_ref = 20   # Reference ambient temp [°C]

        T_cell = T_air + ((noct - T_ref) / G_ref) * GHI

    elif model == "Sandia":
        # Sandia empirical model (with wind)
        a = -3.56  # Wind coefficient
        b = 0.943  # Irradiance coefficient

        # Optionally adjust these if you want tech-specific
        T_cell = T_air + (GHI / 1000) * np.exp(a + b * wind_speed)

    else:
        raise ValueError("Unsupported model. Choose 'NOCT' or 'Sandia'.")

    return T_cell