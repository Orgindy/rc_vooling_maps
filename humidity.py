import numpy as np
import pandas as pd

def compute_relative_humidity(T_air, T_dew, units="K"):
    """
    Compute relative humidity (%) from air temp and dew point.
    
    Parameters
    ----------
    T_air : float, np.ndarray, or pd.Series
        Air temperature (Kelvin or Celsius).
    T_dew : float, np.ndarray, or pd.Series
        Dew point temperature (Kelvin or Celsius).
    units : str
        'K' if inputs are Kelvin, 'C' if Celsius.

    Returns
    -------
    RH : same type as input, clipped 0–100
    """
    T_air = np.array(T_air)
    T_dew = np.array(T_dew)

    if units == "K":
        T_air -= 273.15
        T_dew -= 273.15
    elif units != "C":
        raise ValueError("units must be 'K' or 'C'")

    a = 17.625
    b = 243.04

    e_s = np.exp((a * T_air) / (b + T_air))
    e_d = np.exp((a * T_dew) / (b + T_dew))
    RH = 100.0 * (e_d / e_s)

    return np.clip(RH, 0, 100)


if __name__ == "__main__":
    rh = compute_relative_humidity(293.15, 283.15, units="K")
    print(f"RH (Kelvin input): {rh:.2f} %")

    rh2 = compute_relative_humidity(20.0, 10.0, units="C")
    print(f"RH (Celsius input): {rh2:.2f} %")
