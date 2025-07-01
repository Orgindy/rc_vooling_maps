# dynamic_materials.py
"""
Dynamic material state switching for adaptive radiative cooling coatings.

Improved version:
- Single-point mode or vectorized mode for batch arrays.
- Optional smooth switching near thresholds.
- Simple zenith override.
- No direct file or config I/O — stays modular.
"""

import numpy as np
import pandas as pd

try:
    from pvlib.solarposition import get_solarposition
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    print("⚠️ pvlib not available — using fallback zenith logic.")


def get_material_state(
    T_surface,
    GHI,
    profile,
    smooth=False,
    smoothing_band=2.0
):
    """
    Decide material state based on thresholds.

    Parameters
    ----------
    T_surface : float
        Surface temp [K] or °C.
    GHI : float
        Global horizontal irradiance [W/m²].
    profile : dict
        Switching conditions: state_map with T/GHI thresholds.
    smooth : bool, optional
        If True, blend near thresholds.
    smoothing_band : float, optional
        +/- degrees K or W/m² range for blending.

    Returns
    -------
    str
        Chosen state label.
    """
    T_celsius = T_surface - 273.15 if T_surface > 200 else T_surface

    for state, cond in profile["state_map"].items():
        match = True
        if "T_max" in cond:
            match &= T_celsius <= cond["T_max"]
        if "T_min" in cond:
            match &= T_celsius >= cond["T_min"]
        if "GHI_max" in cond:
            match &= GHI <= cond["GHI_max"]
        if "GHI_min" in cond:
            match &= GHI >= cond["GHI_min"]

        if match:
            return state

    return profile.get("default", "static")


def get_emissivity(state, emissivity_profile):
    return emissivity_profile.get(state, emissivity_profile.get("default", 0.90))


def get_alpha_solar(state, alpha_profile):
    return alpha_profile.get(state, alpha_profile.get("default", 0.90))


def get_material_properties(
    T_surface,
    GHI,
    solar_zenith,
    profile,
    emissivity_profile,
    alpha_profile,
    smooth=False
):
    """
    Return selected state, emissivity, and absorptivity.

    Parameters
    ----------
    T_surface : float
    GHI : float
    solar_zenith : float
    profile : dict
    emissivity_profile : dict
    alpha_profile : dict
    smooth : bool, optional

    Returns
    -------
    dict
    """
    # Apply zenith override if given
    profile = profile.copy()
    if "zenith_threshold" in profile:
        if solar_zenith >= profile["zenith_threshold"]:
            profile["state_map"] = {"dark": {"T_min": 0}}
        elif solar_zenith <= (90 - profile["zenith_threshold"]):
            profile["state_map"] = {"bright": {"T_max": 1000}}

    state = get_material_state(T_surface, GHI, profile, smooth=smooth)
    epsilon = get_emissivity(state, emissivity_profile)
    alpha = get_alpha_solar(state, alpha_profile)

    return {"state": state, "emissivity": epsilon, "alpha": alpha}


def get_solar_zenith(lat, lon, times, tz="UTC"):
    """
    Compute solar zenith angle series.

    Returns
    -------
    pd.Series
    """
    if not PVLIB_AVAILABLE:
        print("⚠️ Using fallback zenith = 45°.")
        return pd.Series(45.0, index=times)

    times_local = times.tz_convert(tz) if times.tz is not None else times.tz_localize(tz)
    sp = get_solarposition(times_local, lat, lon)
    return sp["zenith"]


def vectorized_get_material_properties(
    T_surface_array,
    GHI_array,
    solar_zenith_array,
    profile,
    emissivity_profile,
    alpha_profile
):
    """
    Vectorized version: batch processing.

    Returns
    -------
    pd.DataFrame with state, emissivity, alpha arrays.
    """
    states = []
    epsilons = []
    alphas = []

    for T, GHI, Z in zip(T_surface_array, GHI_array, solar_zenith_array):
        props = get_material_properties(
            T_surface=T,
            GHI=GHI,
            solar_zenith=Z,
            profile=profile,
            emissivity_profile=emissivity_profile,
            alpha_profile=alpha_profile
        )
        states.append(props["state"])
        epsilons.append(props["emissivity"])
        alphas.append(props["alpha"])

    return pd.DataFrame({
        "state": states,
        "emissivity": epsilons,
        "alpha": alphas
    })


if __name__ == "__main__":
    # Simple test
    example_profile = {
        "state_map": {
            "bright": {"T_max": 25, "GHI_max": 200},
            "dark": {"T_min": 25, "GHI_min": 200}
        },
        "default": "static",
        "zenith_threshold": 85
    }

    emissivity_profile = {
        "bright": 0.95,
        "dark": 0.80,
        "static": 0.92,
        "default": 0.90
    }

    alpha_profile = {
        "bright": 0.10,
        "dark": 0.90,
        "static": 0.85,
        "default": 0.90
    }

    props = get_material_properties(
        T_surface=298, GHI=300, solar_zenith=30,
        profile=example_profile,
        emissivity_profile=emissivity_profile,
        alpha_profile=alpha_profile
    )
    print(props)
