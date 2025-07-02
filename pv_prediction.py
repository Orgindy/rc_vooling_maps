import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from plot_utils import apply_standard_plot_style, save_figure
from pathlib import Path
import joblib
from datetime import datetime
import os
from config import get_path
from xgboost import XGBRegressor
import logging
from pv_profiles import pv_profiles as default_pv_profiles
import argparse
from utils.feature_utils import (
    compute_band_ratios,
    filter_valid_columns,
    compute_cluster_spectra,
)
from sklearn.gaussian_process.kernels import RBF

def validate_file_readable(path, binary=False):
    """Return True if file exists and can be opened (text or binary)."""
    if not os.path.isfile(path):
        logging.error(f"File not found: {path}")
        return False
    try:
        mode = "rb" if binary else "r"
        with open(path, mode):
            pass
    except Exception as exc:
        logging.error(f"Cannot read file {path}: {exc}")
        return False
    return True

# -----------------------------
# PV Cell Profile Management
# -----------------------------

def load_pv_profiles_from_csv(file_path=None):
    from config import get_path

    if file_path is None:
        file_path = get_path("pv_profile_path")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Profile file not found: {file_path}")
    
    df = pd.read_csv(file_path)

    # Validate required columns
    required_cols = {'Technology', 'Blue', 'Green', 'Red', 'IR', 'TempCoeff'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file missing required columns: {required_cols - set(df.columns)}")

    spectral_cols = ['Blue', 'Green', 'Red', 'IR']
    df = df.set_index('Technology')
    profiles = {
        tech: {
            'spectral_response': row[spectral_cols].to_dict(),
            'temperature_coefficient': row['TempCoeff']
        }
        for tech, row in df.to_dict(orient='index').items()
    }
    return profiles

def get_pv_cell_profiles():
    logging.info("Using internal default PV profiles from module.")
    return default_pv_profiles

def prepare_features_for_ml(df):
    base_features = [
        'GHI', 'T_air', 'RC_potential', 'Wind_Speed',
        'Dew_Point', 'Cloud_Cover', 'Red_band',
        'Blue_band', 'IR_band', 'Total_band'
    ]

    df = df.copy()
    df, ratio_cols = compute_band_ratios(
        df,
        ['Blue_band', 'Green_band', 'Red_band', 'IR_band'],
        total_col='Total_band'
    )
    feature_names = base_features + ratio_cols

    X = filter_valid_columns(df, feature_names)
    y = df['PV_Potential']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, list(X.columns), scaler

def train_random_forest(
    X_scaled,
    y,
    feature_names,
    test_size=0.2,
    random_state=42,
    n_estimators=200,
    max_depth=12,
    n_clusters=5,
    output_plot=None,
    model_dir=os.path.join(get_path("results_path"), "models"),
    top_k_features=10  # Optional: limit feature importance plot
):
    """
    Train a Random Forest model and save the model and feature importance plot.

    Returns:
        model, X_train, X_test, y_train, y_test, y_pred, model_path, metrics_dict
    """
    # --- Validate inputs ---
    if X_scaled.shape[1] != len(feature_names):
        raise ValueError(f"Feature mismatch: X has {X_scaled.shape[1]} features, but {len(feature_names)} names provided.")

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    # --- Train model ---
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Evaluation ---
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"R² Score: {r2:.4f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"MAE: {mae:.2f}")

    # --- Save model ---
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(model_dir, f"rf_model_k{n_clusters}_{timestamp}.joblib")
    joblib.dump(model, model_path)
    logging.info(f"💾 Random Forest model saved to: {model_path}")

    # --- Plot and save feature importance ---
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    if top_k_features:
        feature_importance_df = feature_importance_df.head(top_k_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    apply_standard_plot_style(
        ax,
        title="Feature Importance (Random Forest)",
        xlabel="Importance",
        ylabel="Feature",
    )

    if output_plot:
        save_figure(fig, os.path.basename(output_plot), folder=os.path.dirname(output_plot) or '.')
        logging.info(f"📊 Feature importance plot saved to: {output_plot}")
    else:
        plt.show()

    plt.close()

    # --- Return everything, including metrics ---
    metrics = {"R2": r2, "RMSE": rmse, "MAE": mae}
    return model, X_train, X_test, y_train, y_test, y_pred, model_path, metrics

def train_ensemble_model(
    df,
    feature_cols,
    target_col='PV_Potential_physics',
    test_size=0.2,
    random_state=42,
    log_scores=False
):
    """
    Train ensemble of RF, XGBoost, and GPR, return predictions and GPR uncertainty.
    """
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in input DataFrame: {missing_cols}")

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train Models
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=random_state)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=random_state)
    xgb.fit(X_train, y_train)

    gpr = GaussianProcessRegressor(kernel=RBF(), alpha=1e-2, normalize_y=True)
    gpr.fit(X_train, y_train)

    # Predict on full dataset
    rf_preds = rf.predict(X)
    xgb_preds = xgb.predict(X)
    gpr_preds, gpr_std = gpr.predict(X, return_std=True)

    # Ensemble prediction (equal weights)
    ensemble_preds = (rf_preds + xgb_preds + gpr_preds) / 3

    # Attach predictions
    df_out = df.copy()
    df_out['Predicted_PV_Potential'] = ensemble_preds
    df_out['Prediction_Uncertainty'] = gpr_std

    if log_scores:
        logging.info("\n=== Ensemble Model Evaluation ===")
        logging.info(f"RF R²: {r2_score(y, rf_preds):.4f}")
        logging.info(f"XGB R²: {r2_score(y, xgb_preds):.4f}")
        logging.info(f"GPR R²: {r2_score(y, gpr_preds):.4f}")

    logging.info(f"✅ Ensemble model trained with {len(feature_cols)} features")
    return df_out

def predict_pv_potential(model, X_scaled, df_original):
    predictions = model.predict(X_scaled)
    df_result = df_original.copy()
    df_result['Predicted_PV_Potential'] = predictions
    return df_result

def train_hybrid_ml_models(X, y):
    """
    Trains multiple ML models and compares their performance.
    
    Parameters:
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Target variable (PV potential).
    
    Returns:
    - models (dict): Trained models with their scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # Gaussian Process Regressor
    gpr = GaussianProcessRegressor()
    gpr.fit(X_train, y_train)
    gpr_pred = gpr.predict(X_test)
    
    # Model performance
    models = {
        "RandomForest": {
            "model": rf,
            "R2": r2_score(y_test, rf_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred))
        },
        "GaussianProcess": {
            "model": gpr,
            "R2": r2_score(y_test, gpr_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, gpr_pred))
        }
    }
    
    logging.info("\n=== Model Performance ===")
    for name, data in models.items():
        logging.info(f"{name} - R²: {data['R2']:.4f}, RMSE: {data['RMSE']:.2f}")
    
    return models

def run_kmedoids_clustering(X_scaled, n_clusters=4, metric='euclidean', random_state=42):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, init='k-medoids++', random_state=random_state)
    kmedoids.fit(X_scaled)
    labels = kmedoids.labels_
    silhouette = silhouette_score(X_scaled, labels)
    logging.info(f"K-Medoids Silhouette Score: {silhouette:.4f}")
    return kmedoids, labels, silhouette


def evaluate_cluster_quality(X_scaled, cluster_labels):
    """
    Evaluates clustering performance using multiple metrics.
    
    Parameters:
    - X_scaled (np.array): Standardized feature matrix used for clustering.
    - cluster_labels (np.array): Labels assigned to each sample by the clustering algorithm.
    
    Returns:
    - scores (dict): Dictionary of metric names and their values.
    """
    try:
        silhouette = silhouette_score(X_scaled, cluster_labels)
        calinski = calinski_harabasz_score(X_scaled, cluster_labels)
        davies = davies_bouldin_score(X_scaled, cluster_labels)

        scores = {
            "Silhouette Score": round(silhouette, 4),
            "Calinski-Harabasz Score": round(calinski, 2),
            "Davies-Bouldin Score": round(davies, 4)
        }

        logging.info("\n=== Clustering Quality Metrics ===")
        for k, v in scores.items():
            logging.info(f"{k}: {v}")

        return scores

    except Exception as e:
        logging.warning(f"❌ Cluster evaluation failed: {e}")
        return {}


def assign_clusters_to_dataframe(df, labels, column_name='Cluster_ID'):
    df_out = df.copy()
    df_out[column_name] = labels
    return df_out

# -----------------------------
# PV Technology Matching
# -----------------------------


def match_technology_to_clusters(cluster_spectra_df, pv_profiles, temp_col='T_air'):
    spectral_bands = ['Blue', 'Green', 'Red', 'IR']
    band_cols = [f'{b}_band' for b in spectral_bands]

    tech_names = list(pv_profiles)
    coeffs = np.array([
        [pv_profiles[t]['spectral_response'][b] for b in spectral_bands]
        for t in tech_names
    ])
    temp_coeffs = np.array([pv_profiles[t]['temperature_coefficient'] for t in tech_names])

    cluster_vals = cluster_spectra_df[band_cols].values
    scores = cluster_vals @ coeffs.T
    temp_penalty = (cluster_spectra_df[temp_col].values[:, None] - 25) * temp_coeffs
    total_scores = scores + temp_penalty

    best_idx = total_scores.argmax(axis=1)
    match_df = pd.DataFrame(total_scores, columns=tech_names)
    match_df['Cluster_ID'] = cluster_spectra_df['Cluster_ID'].values
    match_df['Best_Technology'] = [tech_names[i] for i in best_idx]

    cols = ['Cluster_ID', 'Best_Technology'] + tech_names
    return match_df[cols]


def plot_clusters_map(df, lat_col='latitude', lon_col='longitude', cluster_col='Cluster_ID', title='PV Performance Clusters'):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326").to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(ax=ax, column=cluster_col, cmap='tab10', legend=True, markersize=35, edgecolor='k')
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception:
        logging.warning("Basemap could not be loaded.")
    ax.set_axis_off()
    apply_standard_plot_style(ax, title=title)
    plt.show()


def prepare_features_for_clustering(df, feature_cols):
    """
    Extract and standardize the feature space for clustering.

    Parameters:
    - df: DataFrame containing input features
    - feature_cols: list of column names to use for clustering

    Returns:
    - X_scaled: standardized numpy array of features
    - valid_idx: index of rows that passed filtering (for assigning cluster labels back)
    """
    df_features = df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    return X_scaled, df_features.index



def multi_year_clustering(
    input_dir=Path(get_path("results_path")) / "merged_years",
    output_dir=Path(get_path("results_path")) / "clustered_outputs",
    n_clusters=5,
    file_pattern="merged_dataset_*.nc",
):
    """
    Run main_clustering_pipeline across multiple years of NetCDF datasets.

    Parameters:
    - input_dir: Path to directory with merged yearly .nc files
    - output_dir: Path to save clustered & matched outputs
    - n_clusters: number of clusters for K-Medoids
    - file_pattern: filename pattern for yearly NetCDFs

    Returns:
    - summary_df: Combined DataFrame with cluster+technology assignments per year
    """
    import xarray as xr
    import pandas as pd

    input_dir = Path(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob(file_pattern))
    if not input_files:
        logging.warning("⚠️ No NetCDF files found for pattern.")
        return

    matched_tech_dfs = []

    for file in input_files:
        year = ''.join(filter(str.isdigit, file.stem))  # Extracts digits from filename
        logging.info(f"\n📅 Processing year: {year} — {file.name}")

        # Output filenames to look for after clustering
        matched_out = output_dir / f'matched_dataset_{year}.csv'

        df_clustered = main_clustering_pipeline(
            input_file=str(file),
            output_dir=str(output_dir),
            n_clusters=n_clusters
        )

        # Post-processing: append matched dataset
        if not matched_out.exists():
            raise FileNotFoundError(f"Expected matched dataset not found: {matched_out}")
        match_df = pd.read_csv(matched_out)
        match_df['Year'] = year
        matched_tech_dfs.append(match_df)

    # Combine all years
    summary_df = pd.concat(matched_tech_dfs, ignore_index=True)
    summary_csv = output_dir / 'summary_technology_matching.csv'
    summary_df.to_csv(summary_csv, index=False)

    logging.info(f"\n✅ Multi-year clustering completed. Summary saved: {summary_csv}")
    return summary_df

def generate_cluster_summaries(clustered_df, cluster_col='Cluster_ID', save_path=None):
    """
    Generate summaries per cluster: stats, spectral profiles, technology distributions.

    Parameters:
    - clustered_df: DataFrame with clustered and matched PV technology data.
    - cluster_col: name of column containing cluster labels.
    - save_path: if provided, saves summary CSV to this path.

    Returns:
    - summary_df: summary DataFrame.
    """
    summary_stats = clustered_df.groupby(cluster_col).agg({
        'GHI': 'mean',
        'T_air': 'mean',
        'RC_potential': 'mean',
        'Red_band': 'mean',
        'Blue_band': 'mean',
        'IR_band': 'mean',
        'Total_band': 'mean',
        'PV_Potential_physics': 'mean',
        'Predicted_PV_Potential': 'mean',
        'latitude': ['mean', 'std'],
        'longitude': ['mean', 'std']
    })

    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats.reset_index(inplace=True)

    # Count points per cluster
    cluster_sizes = clustered_df[cluster_col].value_counts().reset_index()
    cluster_sizes.columns = [cluster_col, 'n_points']
    summary_df = summary_stats.merge(cluster_sizes, on=cluster_col)

    # Most common technology per cluster
    if 'Best_Technology' in clustered_df.columns:
        top_tech = clustered_df.groupby(cluster_col)['Best_Technology'] \
                               .agg(lambda x: x.value_counts().idxmax()) \
                               .reset_index(name='Dominant_Technology')
        summary_df = summary_df.merge(top_tech, on=cluster_col)

    if save_path:
        summary_df.to_csv(save_path, index=False)
        logging.info(f"✅ Cluster summary saved to: {save_path}")

    return summary_df

def generate_zone_descriptions(df, cluster_col='Cluster_ID'):
    grouped = df.groupby(cluster_col).agg({
        'T_air': 'mean',
        'RC_potential': 'mean',
        'Red_band': 'mean',
        'Best_Technology': lambda x: x.mode().iloc[0]
    })

    def describe_row(row):
        desc = []
        desc.append("Hot" if row['T_air'] > 18 else "Cool")
        desc.append("High RC" if row['RC_potential'] > 40 else "Low RC")
        desc.append("Red-rich" if row['Red_band'] > 0.35 else "Red-poor")
        desc.append(f"→ {row['Best_Technology']}")
        return " ".join(desc)

    grouped['Zone_Description'] = grouped.apply(describe_row, axis=1)
    return grouped[['Zone_Description']]


def summarize_and_plot_multi_year_clusters(summary_df, output_dir=os.path.join(get_path('results_path'), 'clusters')):
    """
    Generate seasonal + yearly summaries and cluster maps.
    Saves summary CSVs and plots per year.

    Parameters:
    - summary_df: combined cluster-to-technology DataFrame with 'Year' column.
    - output_dir: location to save visualizations and summaries.
    """
    output_dir = Path(output_dir)

    # Plot best technology frequency per year
    tech_freq = summary_df.groupby(['Year', 'Best_Technology']).size().reset_index(name='Count')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=tech_freq, x='Year', y='Count', hue='Best_Technology', ax=ax)
    apply_standard_plot_style(
        ax,
        title="Best PV Technology Frequency per Year",
        xlabel="Year",
        ylabel="Count",
    )
    save_figure(fig, 'tech_frequency_per_year.png', folder=output_dir)

    # Compute basic yearly averages per tech
    avg_scores = summary_df.groupby(['Year', 'Best_Technology']).mean(numeric_only=True).reset_index()
    avg_scores.to_csv(output_dir / 'average_scores_per_tech_year.csv', index=False)

    logging.info(f"📊 Saved summary plots and CSVs to {output_dir}")

def rc_only_clustering(df, n_clusters=5):
    from sklearn_extra.cluster import KMedoids
    from sklearn.preprocessing import StandardScaler

    features = ['P_rc_net', 'T_air', 'RH', 'Wind_Speed']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    labels = kmedoids.fit_predict(X_scaled)
    df['RC_Cluster'] = -1
    df.loc[X.index, 'RC_Cluster'] = labels

    return df, kmedoids

def main_clustering_pipeline(
    input_file=get_path('merged_data_path'),
    output_dir=get_path('results_path'),
    n_clusters=5,
):
    import xarray as xr

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found or unreadable: {input_file}")

    logging.info(f"📥 Loading dataset: {input_file}")
    ds = xr.open_dataset(input_file)
    df = ds.to_dataframe().reset_index()
    df = df.dropna(subset=['GHI', 'T_air', 'RC_potential', 'Red_band', 'Total_band'])

    # --- Timestamp and folders ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    cluster_dir = os.path.join(output_dir, "clusters")
    plot_dir = os.path.join(output_dir, "plots")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(cluster_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- File paths ---
    output_clustered = os.path.join(cluster_dir, f"clustered_dataset_{run_id}.csv")
    output_matched = os.path.join(cluster_dir, f"matched_dataset_{run_id}.csv")
    output_plot = os.path.join(plot_dir, f"feature_importance_{run_id}.png")
    output_summary = os.path.join(cluster_dir, f"cluster_summary_{run_id}.csv")
    output_zones = os.path.join(cluster_dir, f"zone_descriptions_{run_id}.csv")
    model_path = os.path.join(model_dir, f"rf_model_k{n_clusters}_{run_id}.joblib")
    metrics_path = os.path.join(cluster_dir, f"model_metrics_k{n_clusters}_{run_id}.txt")
    sil_path = os.path.join(cluster_dir, f"silhouette_score_{n_clusters}_{run_id}.txt")

    # --- Feature preparation ---
    logging.info("🔧 Preparing features for Random Forest...")
    X_scaled, y, feature_names, scaler_rf = prepare_features_for_ml(df)

    logging.info("🧠 Training Random Forest...")
    model, X_train, X_test, y_train, y_test, y_pred, model_path = train_random_forest(
        X_scaled,
        y.values,
        feature_names,
        n_clusters=n_clusters,
        output_plot=output_plot,
        model_dir=model_dir
    )

    with open(metrics_path, 'w') as f:
        f.write("=== Random Forest Model Evaluation ===\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"n_clusters: {n_clusters}\n\n")
        f.write(f"R² Score: {r2_score(y_test, y_pred):.4f}\n")
        f.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}\n")
        f.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}\n")
    logging.info(f"📄 Model evaluation metrics saved to: {metrics_path}")

    # --- Predict PV Potential ---
    df_with_pred = predict_pv_potential(model, X_scaled, df)
    df = df_with_pred.copy()

    # --- Ensemble Model ---
    logging.info("🎯 Training ensemble model...")
    ensemble_features = ['GHI', 'T_air', 'RC_potential', 'Wind_Speed', 'Dew_Point',
                         'Cloud_Cover', 'Blue_band', 'Red_band', 'IR_band', 'Total_band']
    df_with_pred = train_ensemble_model(df_with_pred, ensemble_features)

    # --- Clustering ---
    logging.info("🔍 Preparing features for clustering...")
    clustering_features = ['GHI', 'T_air', 'RC_potential', 'Wind_Speed', 'Dew_Point',
                           'Blue_band', 'Red_band', 'IR_band', 'Total_band', 'Predicted_PV_Potential']
    X_cluster_scaled, valid_idx = prepare_features_for_clustering(df_with_pred, clustering_features)

    logging.info("🔗 Running K-Medoids clustering...")
    kmedoids, labels, silhouette = run_kmedoids_clustering(X_cluster_scaled, n_clusters=n_clusters)
    with open(sil_path, 'w') as f:
        f.write(f"Silhouette Score: {silhouette:.4f}\n")
    logging.info(f"📄 Silhouette score saved to: {sil_path}")

    df_clustered = assign_clusters_to_dataframe(df_with_pred, labels)

    # --- Spectrum Matching ---
    logging.info("📊 Computing Cluster-Averaged Spectra and Temperatures...")
    cluster_spectra = compute_cluster_spectra(df_clustered, cluster_col='Cluster_ID')

    logging.info("🔬 Matching PV Technologies to Clusters...")
    pv_profiles = get_pv_cell_profiles()
    match_df = match_technology_to_clusters(cluster_spectra, pv_profiles)
    match_df.to_csv(output_matched, index=False)
    logging.info(f"✅ Technology-matched dataset saved to: {output_matched}")

    df_clustered.to_csv(output_clustered, index=False)
    logging.info(f"✅ Clustered dataset saved to: {output_clustered}")

    # --- Summarize Clusters ---
    logging.info("📈 Generating cluster summaries...")
    summary_df = generate_cluster_summaries(df_clustered, save_path=output_summary)

    logging.info("🗺️ Generating zone descriptions...")
    zone_descriptions = generate_zone_descriptions(df_clustered)
    zone_descriptions.to_csv(output_zones)
    logging.info(f"✅ Zone descriptions saved to: {output_zones}")

    # --- Visualization ---
    logging.info("🗺️ Plotting clusters on map...")
    plot_clusters_map(df_clustered)

    return df_clustered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PV clustering pipeline on NetCDF data")
    parser.add_argument("--input-dir", default=os.path.join(get_path("results_path"), "merged_years"))
    parser.add_argument("--output-dir", default=os.path.join(get_path("results_path"), "clusters"))
    parser.add_argument("--file-pattern", default="*.nc")
    parser.add_argument("--n-clusters", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    file_pattern = args.file_pattern

    nc_files = sorted(input_dir.glob(file_pattern))
    if not nc_files:
        logging.error(f"❌ No NetCDF files found in {input_dir} matching '{file_pattern}'")
        exit(1)

    all_matches = []

    for nc_file in nc_files:
        year = nc_file.stem.replace("merged_", "").replace(".nc", "")
        logging.info(f"\n📅 Processing file for year {year}: {nc_file.name}")

        df_clustered = main_clustering_pipeline(
            input_file=str(nc_file),
            output_dir=str(output_dir),
            n_clusters=args.n_clusters
        )

        match_path = output_dir / "clusters" / f"matched_dataset_{year}.csv"
        if match_path.exists():
            match_df = pd.read_csv(match_path)
            match_df["Year"] = year
            all_matches.append(match_df)

    if all_matches:
        summary_df = pd.concat(all_matches, ignore_index=True)
        summary_path = output_dir / "summary_technology_matching.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"\n✅ Summary of technology matches saved to: {summary_path}")
        summarize_and_plot_multi_year_clusters(summary_df, output_dir=str(output_dir))