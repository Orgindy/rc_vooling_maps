"""Central configuration for physical constants and parameters."""

#Sample Integration Example
#from pv_profiles import PV_CONSTANTS
#coeff = PV_CONSTANTS["temperature_coefficients"].get(tech_type, default_val)

RC_MATERIALS = {
    "Default_Coating": {
        "alpha_solar": 0.85,
        "epsilon_IR": 0.92,
        "thickness_m": 0.003,
        "density": 2500,
        "cp": 900,
        "h_conv_base": 5,
        "h_conv_wind_coeff": 4,
    },
    "Smart_Coating": {
        "alpha_solar": 0.85,
        "epsilon_IR": 0.95,
        "thickness_m": 0.002,
        "density": 2400,
        "cp": 880,
        "h_conv_base": 5,
        "h_conv_wind_coeff": 4,
        "switching_profile": {
            "state_map": {
                "bright": {"T_max": 25, "GHI_max": 200},
                "dark": {"T_min": 25, "GHI_min": 200}
            },
            "default": "static",
            "zenith_threshold": 85
        },
        "emissivity_profile": {
            "bright": 0.95,
            "dark": 0.80,
            "static": 0.92,
            "default": 0.90
        },
        "alpha_profile": {
            "bright": 0.10,
            "dark": 0.90,
            "static": 0.85,
            "default": 0.90
        }
    }
}

# --- PV Constants ---
PV_CONSTANTS = {
    "temperature_coefficients": {
        "Silicon": -0.0045,
        "Perovskite": -0.0025,
        "Tandem": -0.0035,
        "CdTe": -0.0028,
        "CIGS": -0.0029,
        "a-Si": -0.0016,
        "OPV": -0.0020,
        "DSSC": -0.0015,
        "J-Aggregate": -0.0022,  # experimental
        "Photonics-Enhanced": -0.0030  # lab-based Si/metasurface
    },
    
    "STC_efficiency": {
        "Silicon": 0.20,
        "Perovskite": 0.18,
        "Tandem": 0.28,
        "CdTe": 0.17,
        "CIGS": 0.19,
        "a-Si": 0.10,
        "OPV": 0.08,
        "DSSC": 0.11,
        "J-Aggregate": 0.09,
        "Photonics-Enhanced": 0.25
    },

    "degradation_rate": {
        "Silicon": 0.005,
        "Perovskite": 0.02,
        "Tandem": 0.01,
        "CdTe": 0.006,
        "CIGS": 0.007,
        "a-Si": 0.015,
        "OPV": 0.03,
        "DSSC": 0.02,
        "J-Aggregate": 0.025,
        "Photonics-Enhanced": 0.008
    },

    "Reference_Red_Fraction": 0.42,  # AM1.5 red band fraction
    "PR_ref": 0.80,                  # Reference performance ratio
    "PR_bounds": (0.7, 0.9),         # Acceptable range of PR values
    "NOCT": 45                       # Nominal Operating Cell Temperature (°C)
}

def get_pv_cell_profiles():
    """
    Returns a dictionary of PV technology profiles with
    temperature coefficients and nominal efficiency values.

    These profiles are used to estimate PV performance.
    """
    return {
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
    },
    "CIGS": {
        "spectral_response": {"Blue": 0.6, "Green": 0.8, "Red": 1.0, "IR": 0.6},
        "temperature_coefficient": -0.0029
    },
    "a-Si": {
        "spectral_response": {"Blue": 0.9, "Green": 0.8, "Red": 0.6, "IR": 0.1},
        "temperature_coefficient": -0.0016
    },
    "OPV": {
        "spectral_response": {"Blue": 0.7, "Green": 1.0, "Red": 0.4, "IR": 0.1},
        "temperature_coefficient": -0.0020
    },
    "DSSC": {
        "spectral_response": {"Blue": 0.6, "Green": 0.9, "Red": 0.8, "IR": 0.2},
        "temperature_coefficient": -0.0015
    },
    "J-Aggregate": {
        "spectral_response": {"Blue": 0.8, "Green": 1.0, "Red": 0.9, "IR": 0.2},
        "temperature_coefficient": -0.0022
    },
    "Photonics-Enhanced": {
        "spectral_response": {"Blue": 1.1, "Green": 1.1, "Red": 1.1, "IR": 1.1},
        "temperature_coefficient": -0.0030
    }
}


    #--- Spectral & Technology Profiles ---
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
    },
    "CIGS": {
        "spectral_response": {"Blue": 0.6, "Green": 0.8, "Red": 1.0, "IR": 0.6},
        "temperature_coefficient": -0.0029
    },
    "a-Si": {
        "spectral_response": {"Blue": 0.9, "Green": 0.8, "Red": 0.6, "IR": 0.1},
        "temperature_coefficient": -0.0016
    },
    "OPV": {
        "spectral_response": {"Blue": 0.7, "Green": 1.0, "Red": 0.4, "IR": 0.1},
        "temperature_coefficient": -0.0020
    },
    "DSSC": {
        "spectral_response": {"Blue": 0.6, "Green": 0.9, "Red": 0.8, "IR": 0.2},
        "temperature_coefficient": -0.0015
    },
    "J-Aggregate": {
        "spectral_response": {"Blue": 0.8, "Green": 1.0, "Red": 0.9, "IR": 0.2},
        "temperature_coefficient": -0.0022
    },
    "Photonics-Enhanced": {
        "spectral_response": {"Blue": 1.1, "Green": 1.1, "Red": 1.1, "IR": 1.1},
        "temperature_coefficient": -0.0030
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
