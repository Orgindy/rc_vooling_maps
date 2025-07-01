"""Central configuration for physical constants and parameters."""


# --- Physical Limits ---
PHYSICAL_LIMITS = {
    'GHI': (0, 1500),
    'T_air': (-50, 60),
    'RC_potential': (-100, 300),
    'Red_band': (0, None),
    'Total_band': (0, None),
}

# --- Atmospheric Constants ---
ATMOSPHERIC_CONSTANTS = {
    'sigma_sb': 5.670374419e-8,
    'solar_constant': 1361,
    'T_kelvin_offset': 273.15,
}

# --- Humidity & Sky Temp Model Parameters ---
HUMIDITY_LIMITS = {
    'RH': (0, 100),
    'Cloud_Cover': (0, 1),
    'T_dew': (-90, 60),
}

SKY_TEMP_MODEL = {
    'eps_clear_base': 0.741,
    'eps_clear_coeff': 0.0062,
    'emissivity_bounds': (0.7, 1.0),
}