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
    smoothing_band=2.0,
):
    """Vectorized decision of material state.

    Parameters
    ----------
    T_surface : float or np.ndarray
        Surface temperature(s) [K] or °C.
    GHI : float or np.ndarray
        Global horizontal irradiance [W/m²].
    profile : dict
        Switching conditions containing ``state_map`` and optional ``default``.
    smooth : bool, optional
        Currently unused, kept for API compatibility.
    smoothing_band : float, optional
        Unused range for potential blending.

    Returns
    -------
    np.ndarray or str
        Array of state labels matching the shape of ``T_surface``/``GHI``.
    """

    T_surface_arr = np.asarray(T_surface)
    GHI_arr = np.asarray(GHI)
    T_celsius = np.where(T_surface_arr > 200, T_surface_arr - 273.15, T_surface_arr)

    states = np.full(T_celsius.shape, profile.get("default", "static"), dtype=object)
    assigned = np.zeros(T_celsius.shape, dtype=bool)

    for state, cond in profile["state_map"].items():
        mask = np.ones(T_celsius.shape, dtype=bool)
        if "T_max" in cond:
            mask &= T_celsius <= cond["T_max"]
        if "T_min" in cond:
            mask &= T_celsius >= cond["T_min"]
        if "GHI_max" in cond:
            mask &= GHI_arr <= cond["GHI_max"]
        if "GHI_min" in cond:
            mask &= GHI_arr >= cond["GHI_min"]

        mask &= ~assigned
        states[mask] = state
        assigned |= mask

    if states.size == 1:
        return states.item()
    return states


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
    T_arr = np.asarray(T_surface_array)
    GHI_arr = np.asarray(GHI_array)
    Z_arr = np.asarray(solar_zenith_array)

    prof = profile.copy()
    zenith_thr = prof.pop("zenith_threshold", None)

    states = np.full(T_arr.shape, prof.get("default", "static"), dtype=object)
    base_mask = np.ones(T_arr.shape, dtype=bool)

    if zenith_thr is not None:
        dark_mask = Z_arr >= zenith_thr
        bright_mask = Z_arr <= (90 - zenith_thr)
        states[dark_mask] = "dark"
        states[bright_mask] = "bright"
        base_mask = ~(dark_mask | bright_mask)

    if base_mask.any():
        states[base_mask] = get_material_state(
            T_arr[base_mask],
            GHI_arr[base_mask],
            prof,
        )

    vec_eps = np.vectorize(lambda s: emissivity_profile.get(s, emissivity_profile.get("default", 0.90)))
    vec_alpha = np.vectorize(lambda s: alpha_profile.get(s, alpha_profile.get("default", 0.90)))
    epsilons = vec_eps(states)
    alphas = vec_alpha(states)

    return pd.DataFrame({
        "state": states.ravel(),
        "emissivity": epsilons.ravel(),
        "alpha": alphas.ravel(),
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
