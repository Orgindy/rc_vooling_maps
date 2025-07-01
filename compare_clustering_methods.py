# compare_clustering_methods.py
# Purpose: Evaluate and compare multiple clustering methods on the dataset

import argparse
import pandas as pd
import numpy as np
from clustering_methods import run_kmeans, run_gmm, run_dbscan, run_agglomerative
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from config import get_path
from plot_utils import apply_standard_plot_style, save_figure
from sklearn.decomposition import PCA

def load_and_scale_data(csv_path, feature_columns):
    """Load and scale data using the provided feature columns."""
    df = pd.read_csv(csv_path)
    X = df[feature_columns].copy()
    
    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"🔧 Filling {missing_count} missing values with median")
        X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled

def apply_and_store(df, X, method_name, method_func, **kwargs):
    try:
        n_clusters = kwargs.get('n_clusters')
        if n_clusters and X.shape[0] <= n_clusters:
            print(f"⚠️ Not enough samples for {method_name}")
            return (method_name, None)
        labels, score = method_func(X, **kwargs)
        col_name = f"Cluster_{method_name}"
        df[col_name] = labels
        print(f"[{method_name.upper()}] Silhouette Score: {score:.4f}")
        return (method_name, score)
    except Exception as e:
        print(f"Error with method {method_name}: {e}")
        return (method_name, None)

def main():
    parser = argparse.ArgumentParser(description="Compare clustering methods on climate or prediction features.")
    parser.add_argument("--input_csv", type=str, default=os.path.join(get_path("results_path"), "clustered_predicted_pv_rc_combined.csv"),
                        help="Path to input CSV file with features or predictions")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters (used by KMeans, GMM, Agglomerative)")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save output CSV")
    parser.add_argument("--features", nargs="+", default=None, help="List of features to cluster on")
    args = parser.parse_args()

    input_csv = args.input_csv
    n_clusters = args.k
    output_csv = args.output_csv or input_csv.replace(".csv", "_with_methods.csv")
    score_log = input_csv.replace(".csv", "_clustering_scores.csv")

    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return

    df_check = pd.read_csv(input_csv)
    print(f"📊 Available columns: {list(df_check.columns)}")

    potential_features = {
        'GHI': ['GHI', 'SSRD_power', 'Global_Horizontal_Irradiance'],
        'Temperature': ['T_air', 'T2M', 'Temperature'],
        'WindSpeed': ['Wind_Speed', 'WS', 'WindSpeed'],
        'Albedo': ['Albedo', 'fal', 'effective_albedo'],
        'RC_Potential': ['RC_potential', 'P_rc_net', 'QNET', 'Predicted_RC_Cooling'],
        'RH': ['RH', 'Relative_Humidity'],
        'Cloud_Cover': ['Cloud_Cover', 'TCC', 'CloudCover'],
        'Dew_Point': ['Dew_Point', 'TD2M', 'Dewpoint', 'd2m'],
        'Blue_Band': ['Blue_band', 'Blue_Band', 'Blue'],
        'Green_Band': ['Green_band', 'Green_Band', 'Green'],
        'Red_Band': ['Red_band', 'Red_Band', 'Red'],
        'IR_Band': ['IR_band', 'NIR_band', 'IR_Band', 'IR'],
        'Predicted_PV': ['Predicted_PV_Potential']
    }

    feature_columns = []
    feature_map = {}
    if args.features:
        # Use explicitly provided features
        for col in args.features:
            if col in df_check.columns:
                feature_columns.append(col)
            else:
                print(f"⚠️ Requested feature '{col}' not found in dataset")
    else:
        # Auto-detect from potential features
        for feature_name, possible_cols in potential_features.items():
            for col in possible_cols:
                if col in df_check.columns:
                    feature_columns.append(col)
                    feature_map[feature_name] = col
                    print(f"✅ Found {feature_name} as '{col}'")
                    break
            else:
                print(f"⚠️ {feature_name} not found in dataset")

    if len(feature_columns) < 2:
        print(f"❌ Not enough features found ({len(feature_columns)}). Need at least 2 for clustering.")
        return

    print(f"🎯 Using features: {feature_columns}")
    df, X = load_and_scale_data(input_csv, feature_columns)

    scores = []
    print("\n=== Running Clustering Methods ===")

    scores.append(apply_and_store(df, X, "kmeans", run_kmeans, n_clusters=n_clusters))
    scores.append(apply_and_store(df, X, "gmm", run_gmm, n_clusters=n_clusters))
    scores.append(apply_and_store(df, X, "dbscan", run_dbscan, eps=0.5, min_samples=5))
    scores.append(apply_and_store(df, X, "agglomerative", run_agglomerative, n_clusters=n_clusters))

    df.to_csv(output_csv, index=False)
    print(f"✅ Saved clustered dataset to {output_csv}")

    score_df = pd.DataFrame(scores, columns=["Method", "SilhouetteScore"])
    score_df.to_csv(score_log, index=False)
    print(f"✅ Saved clustering scores to {score_log}")

    create_comparison_plots(df, X, scores)

def create_comparison_plots(df, X, scores):
    """Create PCA visualization of clustering results and silhouette score bar chart. Optionally include KG class overlay."""
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, axes = plt.subplots(1, 3 if 'KG_class' in df.columns or 'Koppen_Geiger' in df.columns else 2, figsize=(18, 5))

        # === Plot 1: PCA with clustering overlays ===
        ax1 = axes[0]
        methods = ["kmeans", "gmm", "dbscan", "agglomerative"]
        for method in methods:
            col = f"Cluster_{method}"
            if col in df.columns:
                ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df[col],
                            label=method.upper(), alpha=0.5, s=20, cmap='tab10')
        apply_standard_plot_style(
            ax1,
            title="PCA of Features by Clustering Method",
            xlabel="PC 1",
            ylabel="PC 2"
        )
        ax1.legend(loc='upper right', fontsize=8)

        # === Plot 2: Silhouette scores ===
        ax2 = axes[1]
        method_names = [method for method, _ in scores]
        silhouettes = [score if score is not None else 0 for _, score in scores]
        bars = ax2.bar(method_names, silhouettes, color="skyblue", edgecolor="black")
        ax2.set_ylim(0, max(silhouettes) * 1.1 if max(silhouettes) > 0 else 1)
        apply_standard_plot_style(
            ax2,
            title="Silhouette Scores by Clustering Method",
            xlabel="Method",
            ylabel="Silhouette Score"
        )
        for bar, score in zip(bars, silhouettes):
            if score > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{score:.3f}", ha='center', va='bottom', fontsize=9)

        # === Plot 3 (optional): Köppen–Geiger class ===
        kg_column = 'KG_class' if 'KG_class' in df.columns else ('Koppen_Geiger' if 'Koppen_Geiger' in df.columns else None)
        if kg_column:
            ax3 = axes[2]
            kg_vals = df[kg_column].astype(str).values
            unique_labels = sorted(set(kg_vals))
            color_map = {label: idx for idx, label in enumerate(unique_labels)}
            colors = [color_map[val] for val in kg_vals]
            scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='tab20', s=20, alpha=0.6)
            apply_standard_plot_style(
                ax3,
                title=f"PCA Colored by {kg_column}",
                xlabel="PC 1",
                ylabel="PC 2"
            )
            legend_labels = [f"{label}" for label in unique_labels]
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=label, markerfacecolor=plt.cm.tab20(color_map[label] / len(unique_labels)),
                                  markersize=6) for label in unique_labels]
            ax3.legend(handles=handles, title="KG Class", fontsize=7, loc='best')

        # === Save figure ===
        save_figure(fig, "clustering_comparison_plots.png")
        print("✅ Saved comparison plots: clustering_comparison_plots.png")

    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")

if __name__ == "__main__":
    main()
