# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:21:06 2025

@author: Gindi002
"""
import os
import pandas as pd
import numpy as np
import logging
import warnings
from config import get_path

EMOJI_ENABLED = True

logger = logging.getLogger(__name__)


def calculate_synergy_index(
    T_pv,
    T_rc,
    GHI,
    gamma_pv=-0.004,
    rc_cooling_energy=None,
    normalize_to=None
):
    """
    Calculate PV–RC synergy index.

    Parameters:
        T_pv (np.ndarray): Baseline PV cell temperatures [°C]
        T_rc (np.ndarray): RC-enhanced surface temperatures [°C]
        GHI (np.ndarray): Global horizontal irradiance [W/m²]
        gamma_pv (float): PV temperature coefficient (e.g. -0.004 / °C)
        rc_cooling_energy (np.ndarray or None): Optional RC energy benefit [W/m²]
        normalize_to (float or None): Total insolation or energy for normalization [Wh/m²]

    Returns:
        float: Synergy index [%]
    """
    # Convert to numpy arrays for safe operations
    T_pv = np.array(T_pv)
    T_rc = np.array(T_rc)
    GHI = np.array(GHI)
    
    # Validate inputs
    if len(T_pv) != len(T_rc) or len(T_pv) != len(GHI):
        min_len = min(len(T_pv), len(T_rc), len(GHI))
        warnings.warn(
            "Input arrays have mismatched lengths; truncating to match",
            stacklevel=2,
        )
        logger.warning(
            "Mismatched array lengths (PV=%s, RC=%s, GHI=%s); truncating to %s",
            len(T_pv),
            len(T_rc),
            len(GHI),
            min_len,
        )
        T_pv = T_pv[:min_len]
        T_rc = T_rc[:min_len]
        GHI = GHI[:min_len]
    
    delta_T = T_pv - T_rc  # Cooling benefit [°C]
    # PV temperature coefficient is typically negative (efficiency drops with
    # higher cell temperature). The benefit from radiative cooling should b
    # positive, so we use the absolute value here.
    delta_P = abs(gamma_pv) * delta_T * GHI  # Instantaneous PV power gain [W/m²]

    if rc_cooling_energy is None:
        rc_cooling_energy = np.zeros_like(delta_P)
    else:
        rc_cooling_energy = np.array(rc_cooling_energy)

    synergy = delta_P + rc_cooling_energy  # Total benefit in watts
    synergy = np.nan_to_num(synergy)

    if normalize_to is None:
        normalize_to = np.nansum(GHI)  # Total GHI over the period

    # Avoid division by zero
    if normalize_to == 0:
        return 0.0

    synergy_index = (np.nansum(synergy) / normalize_to) * 100  # %

    return synergy_index


def add_synergy_index(df, gamma_pv=-0.004, rc_energy_col=None):
    """Return DataFrame with a new ``Synergy_Index`` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing ``T_PV``, ``T_RC`` and ``GHI`` columns.
    gamma_pv : float, optional
        PV temperature coefficient. Default ``-0.004``.
    rc_energy_col : str, optional
        Name of a column with additional RC cooling energy.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the new column appended.
    """
    required_cols = ["T_PV", "T_RC", "GHI"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        warnings.warn(
            f"Missing required columns: {missing_cols}", stacklevel=2
        )
        logger.warning("Missing required columns: %s", missing_cols)
        return df

    df = df.copy()
    delta_T = df["T_PV"] - df["T_RC"]
    delta_P = abs(gamma_pv) * delta_T * df["GHI"]

    if rc_energy_col and rc_energy_col in df.columns:
        rc_energy = df[rc_energy_col]
    else:
        rc_energy = 0

    synergy_benefit = delta_P + rc_energy
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            synergy_benefit,
            df["GHI"],
            out=np.zeros_like(synergy_benefit, dtype=float),
            where=df["GHI"] != 0,
        )
    df["Synergy_Index"] = ratio * 100
    return df


def add_synergy_index_to_dataset_vectorized(csv_path, output_path=None, gamma_pv=-0.004, rc_energy_col=None):
    """
    Load a dataset, compute synergy index for each row using vectorized operations, and save updated CSV.

    Parameters:
        csv_path (str): Path to input CSV
        output_path (str): Path to output CSV (default: overwrite input)
        gamma_pv (float): PV temperature coefficient [°C⁻¹]
        rc_energy_col (str or None): Column name for RC energy benefit, if included

    Returns:
        pd.DataFrame: Updated dataframe with 'Synergy_Index' column
    """
    if EMOJI_ENABLED:
        print(f"📥 Loading dataset from {csv_path}")
    else:
        print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    df = add_synergy_index(df, gamma_pv=gamma_pv, rc_energy_col=rc_energy_col)

    if output_path is None:
        output_path = csv_path

    # Ensure output directory exists if a directory component is provided
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    if EMOJI_ENABLED:
        print(f"✅ Synergy index added and saved to: {output_path}")
    else:
        print(f"Synergy index added and saved to: {output_path}")

    return df


def calculate_synergy_metrics_summary(df, group_by_cols=None):
    """
    Calculate summary statistics for synergy index across different groups.
    
    Parameters:
        df (pd.DataFrame): DataFrame with Synergy_Index column
        group_by_cols (list): Columns to group by (e.g., ['Cluster_ID', 'season'])
    
    Returns:
        pd.DataFrame: Summary statistics. When ``group_by_cols`` is ``None`` a
        single-row DataFrame with the descriptive statistics is returned.
    """
    if 'Synergy_Index' not in df.columns:
        warnings.warn(
            "DataFrame must contain 'Synergy_Index' column", stacklevel=2
        )
        logger.warning("DataFrame missing 'Synergy_Index' column")
        return pd.DataFrame()
    
    if group_by_cols:
        summary = df.groupby(group_by_cols)['Synergy_Index'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
    else:
        summary = df['Synergy_Index'].describe().to_frame().T
    
    return summary


if __name__ == "__main__":
    import argparse
    from database_utils import read_table, write_dataframe

    parser = argparse.ArgumentParser(description="Add Synergy_Index to a dataset")
    parser.add_argument(
        "--input",
        default=os.path.join(get_path("results_path"), "clustered_dataset.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(get_path("results_path"), "clustered_dataset_synergy.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"), help="Database URL")
    parser.add_argument("--db-table", default=os.getenv("PV_DB_TABLE", "pv_data"), help="Table name for DB operations")
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji output")
    args = parser.parse_args()

    if args.no_emoji:
        EMOJI_ENABLED = False

    if args.db_url:
        df = read_table(args.db_table, db_url=args.db_url)
        df = add_synergy_index(df)
        write_dataframe(df, args.db_table, db_url=args.db_url, if_exists="replace")
        df.to_csv(args.output, index=False)
        if EMOJI_ENABLED:
            print(f"✅ Results written to DB table {args.db_table}")
        else:
            print(f"Results written to DB table {args.db_table}")
    else:
        add_synergy_index_to_dataset_vectorized(args.input, args.output)
