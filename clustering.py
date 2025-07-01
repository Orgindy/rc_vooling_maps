import os
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import rasterio
from pyproj import Transformer
from config import get_path
from utils.humidity import compute_relative_humidity
from utils.feature_utils import compute_cluster_spectra
from pv_profiles import get_pv_cell_profiles  # Use external definition

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def add_koppen_geiger(df, kg_raster_path, lat_col='latitude', lon_col='longitude'):
    """
    Attach Köppen–Geiger climate classification to each row using a raster TIFF file.

    Parameters:
    - df: DataFrame with latitude and longitude columns
    - kg_raster_path: Path to the raster file
    - lat_col, lon_col: coordinate column names (default: 'latitude', 'longitude')

    Returns:
    - df_out: same DataFrame with 'KG_Code' and 'KG_Label' columns
    """
    df_out = df.copy()

    if not os.path.exists(kg_raster_path):
        logging.warning(f"⚠️ KG raster not found at {kg_raster_path}")
        df_out['KG_Code'] = np.nan
        df_out['KG_Label'] = np.nan
        return df_out

    coords = list(zip(df_out[lon_col], df_out[lat_col]))

    with rasterio.open(kg_raster_path) as src:
        if src.crs and src.crs.to_string() != 'EPSG:4326':
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            coords = [transformer.transform(x, y) for x, y in coords]

        values = [val[0] if val.size > 0 else np.nan for val in src.sample(coords)]

    df_out['KG_Code'] = values

    kg_lookup = {
        1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk',
        8: 'Csa', 9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
        14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
        20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa',
        26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 29: 'ET', 30: 'EF'
    }

    df_out['KG_Label'] = df_out['KG_Code'].map(kg_lookup)
    return df_out

def prepare_clustered_dataset(
    input_path,
    output_path,
    n_clusters=5,
    use_kmedoids=True,
    features_to_use=None,
    scaler=None,
    enrich_with_kg=False,
    kg_raster_path=None
):
    """
    Load ML input data, optionally enrich with Köppen–Geiger classification,
    cluster using specified features and algorithm, and save clustered dataset.

    Parameters:
    - input_path: CSV file with features or predictions (e.g. 'ml_predictions.csv' or raw potential)
    - output_path: path to save the clustered dataset
    - n_clusters: number of clusters
    - use_kmedoids: use KMedoids (True) or KMeans (False)
    - features_to_use: list of feature columns to use for clustering
    - scaler: optional external scaler (default: StandardScaler)
    - enrich_with_kg: add Köppen–Geiger classification (default: False)
    - kg_raster_path: path to Köppen–Geiger raster file

    Returns:
    - df: clustered DataFrame with 'Cluster_ID' and optional KG info
    """
    logging.info(f"📥 Loading data for clustering from: {input_path}")
    if not os.path.exists(input_path):
        logging.error(f"❌ Input file not found: {input_path}")
        return None

    df = pd.read_csv(input_path)
    logging.info(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")

    if enrich_with_kg and kg_raster_path:
        df = add_koppen_geiger(df, kg_raster_path)
        logging.info("🌍 Added Köppen–Geiger classification")

    # Auto-detect features if none provided
    if features_to_use is None:
        features_to_use = [col for col in df.columns if col.startswith("Predicted_")]

    if len(features_to_use) < 2:
        logging.error(f"❌ Not enough features found for clustering: {features_to_use}")
        return None

    X = df[features_to_use].copy()
    if X.isnull().sum().sum() > 0:
        logging.warning("⚠️ Missing values found, filling with column medians")
        X = X.fillna(X.median())

    if scaler is None:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run clustering
    if use_kmedoids:
        from clustering_methods import run_kmedoids_clustering
        model, labels, silhouette = run_kmedoids_clustering(X_scaled, n_clusters=n_clusters)
        logging.info(f"✅ K-Medoids clustering complete (k={n_clusters})")
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        logging.info(f"✅ KMeans clustering complete (k={n_clusters})")

    df["Cluster_ID"] = labels
    df["Silhouette_Score"] = silhouette

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"📦 Clustered dataset saved to: {output_path}")

    return df

def find_optimal_k(X_scaled, k_range=range(2, 11), use_kmedoids=True, random_state=42):
    """
    Evaluate silhouette scores for different k to select best cluster number.

    Parameters:
    - X_scaled: standardized feature matrix
    - k_range: range of k values to test
    - use_kmedoids: whether to use KMedoids or KMeans
    - random_state: reproducibility seed

    Returns:
    - scores: dict mapping k to silhouette score
    - best_k: k with the highest score
    """
    scores = {}

    for k in k_range:
        try:
            if use_kmedoids:
                model = KMedoids(n_clusters=k, init='k-medoids++', random_state=random_state)
            else:
                model = KMeans(n_clusters=k, init='k-means++', random_state=random_state)

            model.fit(X_scaled)
            score = silhouette_score(X_scaled, model.labels_)
            scores[k] = score
            logging.info(f"k = {k}: silhouette = {score:.4f}")

        except Exception as e:
            logging.warning(f"⚠️ Failed for k = {k}: {e}")

    if not scores:
        logging.error("❌ Could not compute silhouette scores for any k.")
        return {}, None

    best_k = max(scores, key=scores.get)
    logging.info(f"✅ Best k: {best_k} (silhouette = {scores[best_k]:.4f})")

    return scores, best_k

def match_technology_to_clusters(cluster_spectra_df, pv_profiles, temp_col='T_air'):
    """
    Match clusters to PV technologies using spectral and temperature suitability.

    Parameters:
    - cluster_spectra_df: DataFrame with spectral averages and temperature per cluster
    - pv_profiles: dict with PV technologies and their spectral + temp coefficients
    - temp_col: column with average cluster temperature

    Returns:
    - match_df: DataFrame with scores and best technology per cluster
    """
    spectral_bands = ['Blue', 'Green', 'Red', 'IR']
    results = []

    for _, row in cluster_spectra_df.iterrows():
        cluster_id = row['Cluster_ID']
        cluster_temp = row[temp_col]
        tech_scores = {}

        for tech, props in pv_profiles.items():
            spectral_score = sum(
                row[f"{band}_band"] * props['spectral_response'][band]
                for band in spectral_bands
            )
            temp_penalty = props['temperature_coefficient'] * (cluster_temp - 25)
            total_score = spectral_score + temp_penalty
            tech_scores[tech] = total_score

        best_tech = max(tech_scores, key=tech_scores.get)
        results.append({
            'Cluster_ID': cluster_id,
            'Best_Technology': best_tech,
            **tech_scores
        })

    return pd.DataFrame(results)

def compute_adjusted_yield_by_technology(df, pv_profiles):
    """
    Estimate adjusted PV yield for each technology at every location.

    Formula:
        Adjusted_Yield = Predicted_PV_Potential × (1 + SpectralMatch + TempGain)

    Parameters:
    - df: DataFrame with location data, predicted PV, and spectral bands
    - pv_profiles: dict with spectral_response and temperature_coefficient

    Returns:
    - DataFrame with [Location_ID, Technology, Adjusted_Yield]
    """
    spectral_bands = ['Blue', 'Green', 'Red', 'IR']
    output = []

    for idx, row in df.iterrows():
        location_id = row.get('location_id', idx)
        T_air = row['T_air']
        pred_pv = row['Predicted_PV_Potential']
        total_band = row['Total_band']

        for tech, props in pv_profiles.items():
            spectral_match = 0
            for band in spectral_bands:
                band_col = f"{band}_band"
                spectral_frac = row[band_col] / total_band if total_band > 0 else 0
                spectral_match += spectral_frac * props['spectral_response'][band]

            temp_gain = props['temperature_coefficient'] * (T_air - 25)
            adjusted_yield = pred_pv * (1 + spectral_match + temp_gain)

            output.append({
                "Location_ID": location_id,
                "Technology": tech,
                "Adjusted_Yield": adjusted_yield
            })

    return pd.DataFrame(output)

def compute_cluster_summary(
    df,
    cluster_col='Cluster_ID',
    output_path=os.path.join(get_path("results_path"), "cluster_summary.csv"),
):
    """
    Compute summary statistics (means) for each cluster and save to CSV.

    Parameters:
    - df: DataFrame with climate, PV, and cluster columns
    - cluster_col: Name of the column holding cluster IDs
    - output_path: CSV output path for summary

    Returns:
    - summary_df: DataFrame with summary stats per cluster
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = df.groupby(cluster_col).agg({
        'T_air': 'mean',
        'RH': 'mean',
        'Albedo': 'mean',
        'Predicted_PV_Potential': 'mean',
        'Cluster_Label': 'first'  # Assumes consistent label within cluster
    }).reset_index()

    summary = summary.rename(columns={
        'T_air': 'Mean_T_air',
        'RH': 'Mean_RH',
        'Albedo': 'Mean_Albedo',
        'Predicted_PV_Potential': 'Mean_PV_Potential'
    })

    summary.to_csv(output_path, index=False)
    logging.info(f"✅ Saved cluster summary to {output_path}")
    return summary


def label_clusters(df, cluster_col='Cluster_ID'):
    """
    Assign human-readable labels to numeric cluster IDs.

    Parameters:
    - df: DataFrame with cluster assignments
    - cluster_col: Column containing numeric cluster labels

    Returns:
    - DataFrame with an added 'Cluster_Label' column
    """
    cluster_label_map = {
        0: "Hot & Sunny",
        1: "Cool & Diffuse",
        2: "High RC & Low Temp",
        3: "Mountainous, Low IR",
        4: "Temperate High PV"
        # Extend or adjust based on k and analysis
    }

    df['Cluster_Label'] = df[cluster_col].map(cluster_label_map).fillna("Unlabeled")
    return df

def compute_pv_potential_by_cluster_year(
    df,
    cluster_col='Cluster_ID',
    year_col='year',
    pv_col='Predicted_PV_Potential',
    output_path=os.path.join(get_path("results_path"), "pv_potential_by_cluster_year.csv"),
):
    """
    Aggregate total and mean PV potential per cluster per year.

    Parameters:
    - df: DataFrame with PV predictions and cluster info
    - cluster_col: name of the cluster column
    - year_col: name of the year column
    - pv_col: name of the predicted PV potential column
    - output_path: where to save the output CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure 'year' exists (derive from 'time' if necessary)
    if year_col not in df.columns:
        if 'time' in df.columns:
            df[year_col] = pd.to_datetime(df['time'], errors='coerce').dt.year
            logging.warning("🕓 Derived year column from 'time'")
        else:
            logging.warning("⚠️ No year or time column found — using year = 0")
            df[year_col] = 0

    grouped = df.groupby([cluster_col, year_col]).agg(
        Total_PV_Potential=(pv_col, 'sum'),
        Mean_PV_Potential=(pv_col, 'mean'),
        Location_Count=(pv_col, 'count')
    ).reset_index()

    grouped.to_csv(output_path, index=False)
    logging.info(f"✅ Saved PV potential by cluster and year to {output_path}")
    return grouped

def add_koppen_geiger_post_clustering(
    df,
    kg_raster_path,
    lat_col='latitude',
    lon_col='longitude'
):
    """
    Add Köppen–Geiger classification labels after clustering (optional enrichment).

    Parameters:
    - df: DataFrame with coordinates
    - kg_raster_path: Path to .tif KG raster
    - lat_col, lon_col: Column names for lat/lon

    Returns:
    - df_out: DataFrame with 'KG_Code' and 'KG_Label'
    """
    if not os.path.exists(kg_raster_path):
        logging.warning(f"⚠️ KG raster not found at {kg_raster_path}")
        df["KG_Code"] = np.nan
        df["KG_Label"] = np.nan
        return df

    coords = list(zip(df[lon_col], df[lat_col]))

    with rasterio.open(kg_raster_path) as src:
        if src.crs.to_string() != 'EPSG:4326':
            transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            coords = [transformer.transform(x, y) for x, y in coords]
        values = [val[0] if val.size > 0 else np.nan for val in src.sample(coords)]

    df["KG_Code"] = values

    kg_lookup = {
        1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk',
        8: 'Csa', 9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
        14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
        20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa',
        26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 29: 'ET', 30: 'EF'
    }

    df["KG_Label"] = df["KG_Code"].map(kg_lookup)
    logging.info("✅ KG classification appended post-clustering.")
    return df

def main():
    from config import TrainingConfig

    logging.info("🚀 Starting clustering pipeline...")

    cfg = TrainingConfig.from_yaml("config.yaml")

    base_dir = get_path("results_path")
    input_files = {
        "with_kg": os.path.join(base_dir, "ml_predictions_kg.csv"),
        "without_kg": os.path.join(base_dir, "ml_predictions.csv")
    }

    cluster_outputs = {}

    for mode, input_file in input_files.items():
        logging.info(f"\n=== Processing: {mode.upper()} ===")

        # Define output file for this mode
        output_file = os.path.join(base_dir, f"clustered_predictions_{mode}.csv")

        # Run clustering on predicted features
        df_clustered = prepare_clustered_dataset(
            input_path=input_file,
            output_path=output_file,
            n_clusters=cfg.n_clusters,
            use_kmedoids=True,
            enrich_with_kg=(mode == "with_kg"),
            kg_raster_path=cfg.kg_raster_path
        )

        if df_clustered is None:
            continue

        # Save clustered version
        df_clustered.to_csv(output_file, index=False)
        logging.info(f"📦 Saved clustered predictions to {output_file}")

        # For versions without KG, append it post-hoc for analysis
        if mode == "without_kg":
            df_with_kg = add_koppen_geiger_post_clustering(df_clustered, cfg.kg_raster_path)
            df_with_kg.to_csv(
                os.path.join(base_dir, "clustered_predictions_without_kg_with_kglabel.csv"),
                index=False
            )
            logging.info("🌍 KG labels appended to 'without_kg' version.")

        cluster_outputs[mode] = df_clustered

    logging.info("✅ Clustering pipeline completed.")

if __name__ == "__main__":
    main()