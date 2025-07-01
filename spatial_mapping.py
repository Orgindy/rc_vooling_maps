import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import logging
import argparse
from config import get_path  # project-specific path manager
# Optional for future use
import plotly.express as px
import warnings
import xarray as xr  # Make sure xr is imported at the top!

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_FONT_SIZE = 14
DEFAULT_MARKER_SIZE = 10
DEFAULT_ALPHA = 0.85
DEFAULT_CRS = "EPSG:4326"
DEFAULT_FIGSIZE = (10, 7)
DEFAULT_BASEMAP = ctx.providers.CartoDB.Positron
DEFAULT_CMAP_CATEGORICAL = "tab10"
DEFAULT_CMAP_CONTINUOUS = "viridis"

def load_dataset(input_path, kg_col=None, eu_only=True):
    """
    Load CSV or NetCDF (.nc) and convert to GeoDataFrame, with optional EU filter and KG overlay.

    Parameters:
    -----------
    input_path : str
        Path to the input CSV or NetCDF file.
    kg_col : str or None
        Name of the Köppen–Geiger classification column (if present).
    eu_only : bool
        If True, filters the data to Europe bounding box.

    Returns:
    --------
    gdf : GeoDataFrame
        GeoDataFrame with Point geometry and relevant features.
    """
    # Load data based on extension
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
        logging.info(f"📄 Loaded CSV: {input_path}")

    elif input_path.endswith(".nc"):
        ds = xr.open_dataset(input_path)
        df = ds.to_dataframe().reset_index()
        logging.info(f"🌐 Loaded NetCDF: {input_path} -> converted to flat DataFrame")

    else:
        raise ValueError("❌ Unsupported input file format. Please provide .csv or .nc")

    # Basic sanity check
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("❌ Input must contain 'latitude' and 'longitude' columns.")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=DEFAULT_CRS
    )

    # Optional EU filter
    if eu_only:
        gdf = gdf[
            (gdf.latitude >= 34) & (gdf.latitude <= 72) &
            (gdf.longitude >= -25) & (gdf.longitude <= 45)
        ]
        logging.info("📍 Applied EU bounding box filter.")

    # Optional KG overlay
    if kg_col and kg_col in gdf.columns:
        gdf[kg_col] = gdf[kg_col].astype(str)
        logging.info(f"🌍 Added KG classification column: {kg_col}")

    return gdf

def plot_static_map(
    gdf,
    column,
    title,
    cmap="tab10",
    save_path=None,
    marker_size=10,
    basemap=True
):
    """
    Plot a GeoDataFrame column as a static map with optional basemap.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geospatial data.
    column : str
        Column to plot (categorical or numeric).
    title : str
        Title of the map.
    cmap : str
        Matplotlib colormap.
    save_path : str or None
        Path to save the image. If None, shows the plot.
    marker_size : int
        Marker size for each point.
    basemap : bool
        If True, adds context basemap.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    gdf.plot(column=column, ax=ax, cmap=cmap, legend=True, markersize=marker_size, alpha=DEFAULT_ALPHA)
    draw_eu_borders(ax, crs=gdf.crs)

    if basemap:
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logging.warning(f"⚠️ Could not add basemap: {e}")

    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logging.info(f"🖼️ Saved map to: {save_path}")
    else:
        plt.show()

    plt.close()

def run_batch_plotting(
    gdf,
    output_dir,
    include_kg=False,
    kg_col="KG_class"
):
    """
    Generate maps for RC clusters, PV clusters, combined clusters, KG class, and technology matches.

    Parameters:
    -----------
    gdf : GeoDataFrame
        The full dataset including clustering and prediction results.
    output_dir : str
        Directory to save all generated maps.
    include_kg : bool
        Whether to generate a KG classification map.
    kg_col : str
        Column name for KG classification.
    """
    os.makedirs(output_dir, exist_ok=True)

    cluster_cols = {
        "RC_Cluster": "Radiative Cooling Zones",
        "PV_Cluster": "PV Potential Zones",
        "Combined_Cluster": "Combined RC + PV Zones"
    }

    for col, title in cluster_cols.items():
        if col in gdf.columns:
            save_path = os.path.join(output_dir, f"{col.lower()}_map.png")
            plot_static_map(
                gdf,
                column=col,
                title=title,
                cmap="tab10" if "Cluster" in col else "Set2",
                save_path=save_path
            )

        # Plot Cooling Degree Days (CDD)
    if "CDD" in gdf.columns:
        save_path = os.path.join(output_dir, "cooling_degree_days_map.png")
        plot_static_map(
            gdf,
            column="CDD",
            title="Cooling Degree Days (CDD)",
            cmap="Reds",
            save_path=save_path
        )

    # Plot Heating Degree Days (HDD)
    if "HDD" in gdf.columns:
        save_path = os.path.join(output_dir, "heating_degree_days_map.png")
        plot_static_map(
            gdf,
            column="HDD",
            title="Heating Degree Days (HDD)",
            cmap="Blues",
            save_path=save_path
        )

    # Plot PV technology match if available
    if "Best_Technology" in gdf.columns:
        save_path = os.path.join(output_dir, "pv_technology_map.png")
        plot_static_map(
            gdf,
            column="Best_Technology",
            title="Best PV Technology by Cluster",
            cmap="Set2",
            save_path=save_path
        )

    # Plot prediction uncertainty if available
    if "Prediction_Uncertainty" in gdf.columns:
        save_path = os.path.join(output_dir, "prediction_uncertainty_map.png")
        plot_static_map(
            gdf,
            column="Prediction_Uncertainty",
            title="PV Prediction Uncertainty",
            cmap="coolwarm",
            save_path=save_path
        )

    # Optional: Plot KG zones
    if include_kg and kg_col in gdf.columns:
        save_path = os.path.join(output_dir, "kg_classification_map.png")
        plot_static_map(
            gdf,
            column=kg_col,
            title="Köppen–Geiger Climate Classification",
            cmap="tab20",
            save_path=save_path
        )
        
        # Plot Net Performance map if available
    if "Net_Performance" in gdf.columns:
        save_path = os.path.join(output_dir, "net_performance_map.png")
        plot_static_map(
            gdf,
            column="Net_Performance",
            title="Net Performance (PV - Cooling Penalty)",
            cmap="viridis",
            save_path=save_path
        )

    # Plot raw PV potential map
    if "PV_Potential" in gdf.columns:
        save_path = os.path.join(output_dir, "pv_potential_map.png")
        plot_static_map(
            gdf,
            column="PV_Potential",
            title="Raw PV Potential",
            cmap="YlOrBr",
            save_path=save_path
        )

    # Plot raw RC cooling map
    if "RC_potential" in gdf.columns:
        save_path = os.path.join(output_dir, "rc_potential_map.png")
        plot_static_map(
            gdf,
            column="RC_potential",
            title="Radiative Cooling Potential",
            cmap="PuBuGn",
            save_path=save_path
        )

def draw_eu_borders(ax, crs="EPSG:4326"):
    # Load world borders and filter EU countries
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    eu_countries = [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland",
        "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
        "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
    ]
    eu = world[world["name"].isin(eu_countries)].to_crs(crs)
    eu.boundary.plot(ax=ax, color="black", linewidth=0.5)

def main():
    parser = argparse.ArgumentParser(description="Generate RC + PV spatial maps from clustered prediction data")
    parser.add_argument("--target", type=str, choices=["rc", "pv", "combined"], required=True,
                        help="Which clustering dataset to visualize: rc, pv, or combined")
    parser.add_argument("--kg", action="store_true", help="Include Köppen–Geiger climate classification map")
    parser.add_argument("--eu", action="store_true", help="Restrict plots to Europe only")
    args = parser.parse_args()

    # Resolve input path from config
    data_paths = get_path("clustered_data_paths")  # expects dict with 'rc', 'pv', 'combined'
    input_csv = data_paths[args.target]

    # Output base directory
    results_dir = get_path("results_path")
    base_map_dir = os.path.join(results_dir, "maps")

    # Load dataset (with full techs)
    gdf = load_dataset(input_csv, kg_col="KG_class" if args.kg else None, eu_only=args.eu)

    # If no 'Tech' column, just run once
    if "Tech" not in gdf.columns:
        subdir = f"{args.target}_maps_with_KG" if args.kg else f"{args.target}_maps"
        map_dir = os.path.join(base_map_dir, subdir)
        run_batch_plotting(gdf, output_dir=map_dir, include_kg=args.kg)
        return

    # Loop over technologies
    for tech in gdf["Tech"].unique():
        gdf_tech = gdf[gdf["Tech"] == tech]
        subdir = f"{tech}_{args.target}_with_KG" if args.kg else f"{tech}_{args.target}"
        map_dir = os.path.join(base_map_dir, subdir)
        logging.info(f"🗺️ Generating maps for technology: {tech}")
        run_batch_plotting(gdf_tech, output_dir=map_dir, include_kg=args.kg)

if __name__ == "__main__":
    main()
