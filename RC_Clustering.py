from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import joblib
import contextily as ctx
from config import get_path
from sklearn.metrics import silhouette_score
import os
from config import get_column, get_path
from utils.sky_temperature import calculate_sky_temperature_improved

def rc_only_clustering(df, features=None, n_clusters=5, cluster_col='RC_Cluster', model_path=None):
    """
    Perform K-Medoids clustering based on RC potential and thermal features.

    Parameters:
    - df: DataFrame with relevant columns
    - features: list of feature columns to use (default: RC + T + wind + RH)
    - n_clusters: number of clusters
    - cluster_col: name of the column to store cluster labels

    Returns:
    - df_out: DataFrame with added cluster column
    - model: trained KMedoids model
    """
    if features is None:
        features = ['P_rc_net', 'T_air', 'RH', 'Wind_Speed']
    
    df_subset = df.dropna(subset=features).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_subset[features])

    model = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean', init='k-medoids++')
    labels = model.fit_predict(X_scaled)
    try:
        score = silhouette_score(X_scaled, labels)
    except Exception:
        score = None
    if score is not None:
        print(f"Silhouette score: {score:.3f}")
    if model_path:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    df_out = df.copy()
    df_out[cluster_col] = -1
    df_out.loc[df_subset.index, cluster_col] = labels

    return df_out, model, score

def plot_overlay_rc_pv_zones(
    df,
    output_path=os.path.join(get_path("results_path"), "maps", "overlay_rc_pv_map.png"),
):
    """
    Overlay RC-only clusters and matched PV technologies on the same map.

    Parameters:
    - df: DataFrame with coordinates, RC cluster, and tech columns
    - output_path: file path to save the output image
    """

    # Dynamically get column names
    rc_col = get_column("rc_cluster")
    tech_col = get_column("pv_tech")
    lat_col = get_column("lat")
    lon_col = get_column("lon")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: RC Clusters (soft alpha circles)
    gdf.plot(ax=ax, column=rc_col, cmap='Pastel1', markersize=50, alpha=0.6,
             legend=True, edgecolor='none', label='RC Cluster')

    # Foreground: PV Technologies (cross markers)
    gdf.plot(ax=ax, column=tech_col, cmap='tab10', markersize=20, marker='x',
             legend=True, edgecolor='black', label='Best Tech')

    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception:
        print("⚠️ Basemap could not be loaded — offline mode.")

    ax.set_title("Overlay of RC Climate Zones and Optimal PV Technologies", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Overlay map saved to: {output_path}")

def generate_zone_descriptions(df):
    """
    Generate human-readable zone descriptions based on cluster characteristics.

    Returns:
    - zone_df: DataFrame with cluster ID and description
    """

    # Dynamically resolve column names
    cluster_col = get_column("cluster_id")
    rc_col = get_column("rc_potential")
    temp_col = get_column("t_air")
    red_col = get_column("red_band")
    tech_col = get_column("pv_tech")

    # Group and summarize
    grouped = df.groupby(cluster_col).agg({
        temp_col: 'mean',
        rc_col: 'mean',
        red_col: 'mean',
        tech_col: lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    }).reset_index()

    # Generate text descriptions
    def describe(row):
        t = row[temp_col]
        rc = row[rc_col]
        red = row[red_col]
        tech = row[tech_col]

        temp_str = "Hot" if t > 18 else "Cool"
        rc_str = "High RC" if rc > 40 else "Low RC"
        red_str = "Red-rich" if red > 0.35 else "Red-poor"

        return f"{temp_str}, {rc_str}, {red_str} → {tech}"

    grouped['Zone_Description'] = grouped.apply(describe, axis=1)
    return grouped[[cluster_col, 'Zone_Description']]

def calculate_rc_power_improved(df, albedo=0.3, emissivity=0.95):
    """
    Compute net radiative cooling power using realistic sky temperature.

    Returns:
    - df_out: DataFrame with columns Q_rad, Q_solar, and P_rc_net
    """

    # Column names from config
    t_air_col = get_column("t_air")         # typically "T_air"
    ghi_col = get_column("ghi")             # typically "GHI"
    rh_col = get_column("rh")               # typically "RH"
    cloud_col = get_column("cloud_cover")   # typically "TCC"

    df_out = df.copy()

    # Fetch values with fallback for missing humidity or clouds
    RH = df.get(rh_col, 50)
    TCC = df.get(cloud_col, 0)
    T_sky = calculate_sky_temperature_improved(df[t_air_col], RH, TCC)

    # Radiative cooling model
    σ = 5.67e-8  # W/m²·K⁴
    T_air_K = df[t_air_col] + 273.15
    T_sky_K = T_sky + 273.15

    df_out['Q_rad'] = emissivity * σ * (T_air_K**4 - T_sky_K**4)
    df_out['Q_solar'] = (1 - albedo) * df[ghi_col]
    df_out['P_rc_net'] = df_out['Q_rad'] - df_out['Q_solar']

    return df_out

def sweep_rc_with_reflectivity(df, albedo_values=None):
    """
    Run RC power calculations across multiple albedo values.

    Returns:
    - combined_df: DataFrame with one row per (location × albedo)
    """
    if albedo_values is None:
        albedo_values = [0.3, 0.6, 1.0]

    results = []
    for alb in albedo_values:
        df_alb = calculate_rc_power_improved(df, albedo=alb)
        df_alb['Albedo'] = alb
        results.append(df_alb)

    return pd.concat(results, ignore_index=True)