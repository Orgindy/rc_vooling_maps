"""Central configuration for physical constants and parameters."""

# --- PV Constants ---
PV_CONSTANTS = {
    'NOCT': 45,
    'PR_ref': 0.80,
    'Reference_Red_Fraction': 0.42,
    'PR_bounds': (0.7, 0.9),
    'temperature_coefficients': {
        'Silicon': -0.0045,
        'Perovskite': -0.0025,
        'Tandem': -0.0035,
        'CdTe': -0.0028,
    },
}

# --- Spectral & Technology Profiles ---
pv_profiles = {
    "Silicon": {
        "spectral_response": {"Blue": 0.3, "Green": 0.7, "Red": 1.0, "IR": 1.0},
        "temperature_coefficient": -0.0045
    },
    "Perovskite": {
        "spectral_response": {"Blue": 1.0, "Green": 0.9, "Red": 0.4, "IR": 0.1},
        "temperature_coefficient": -0.0025
    },
    "Tandem": {
        "spectral_response": {"Blue": 1.0, "Green": 1.0, "Red": 1.0, "IR": 1.0},
        "temperature_coefficient": -0.0035
    },
    "CdTe": {
        "spectral_response": {"Blue": 0.8, "Green": 0.9, "Red": 0.7, "IR": 0.2},
        "temperature_coefficient": -0.0028
    }
}

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