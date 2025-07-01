#!/usr/bin/env python
"""
Standalone script to visualize radiative cooling potential maps for EU area.
Takes yearly and seasonal aggregated CSV files as input and creates maps.

Usage:
    python visualize_rc_maps.py yearly_file.csv seasonal_file.csv output_directory
"""
import os
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from config import get_path, get_column

# Output format (e.g. 'png', 'svg', 'pdf'), defaults to 'png'
OUTPUT_FORMAT = os.getenv("MAP_FORMAT", "png")

def load_rc_data_from_netcdf():
    """
    Load yearly and seasonal RC cooling data from NetCDF files defined in config.

    Returns:
        Tuple of two DataFrames: (yearly_df, seasonal_df)
    """
    yearly_path = get_path("rc_yearly_file")
    seasonal_path = get_path("rc_seasonal_file")

    yearly_df = xr.open_dataset(yearly_path).to_dataframe().reset_index()
    seasonal_df = xr.open_dataset(seasonal_path).to_dataframe().reset_index()

    return yearly_df, seasonal_df


def get_eu_boundaries(df):
    """
    Get latitude and longitude boundaries for the data, focused on EU region.

    Parameters:
        df (pandas.DataFrame): DataFrame with spatial coordinates

    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) boundary coordinates
    """
    lon_col = get_column("lon")
    lat_col = get_column("lat")

    lon_min = df[lon_col].min()
    lon_max = df[lon_col].max()
    lat_min = df[lat_col].min()
    lat_max = df[lat_col].max()

    lon_min_eu = max(-20, lon_min)
    lon_max_eu = min(40, lon_max)
    lat_min_eu = max(35, lat_min)
    lat_max_eu = min(72, lat_max)

    lon_min = lon_min_eu if lon_min_eu >= lon_min else lon_min
    lon_max = lon_max_eu if lon_max_eu <= lon_max else lon_max
    lat_min = lat_min_eu if lat_min_eu >= lat_min else lat_min
    lat_max = lat_max_eu if lat_max_eu <= lat_max else lat_max

    return lon_min, lon_max, lat_min, lat_max

def create_yearly_maps(yearly_df, output_dir, boundaries):
    """
    Create yearly maps for P_rc_basic and P_rc_net.

    Parameters:
        yearly_df (pandas.DataFrame): DataFrame with yearly aggregated data
        output_dir (str): Directory to save output maps
        boundaries (tuple): (lon_min, lon_max, lat_min, lat_max) boundary coordinates
    """
    try:
        lon_min, lon_max, lat_min, lat_max = boundaries

        # Select the most recent year
        latest_year = yearly_df['year'].max()
        df_year = yearly_df[yearly_df['year'] == latest_year]

        # Plot settings
        fig = plt.figure(figsize=(18, 8))
        projection = ccrs.PlateCarree()
        cmap = 'viridis'

        # Get column names from config
        lon_col = get_column("lon")
        lat_col = get_column("lat")
        rc_basic_col = get_column("p_rc_basic")
        rc_net_col = get_column("p_rc_net")
        cluster_col = get_column("cluster_id", optional=True)

        # First subplot — Basic RC
        ax1 = plt.subplot(1, 2, 1, projection=projection)
        vmin_basic = max(0, df_year[rc_basic_col].min())
        vmax_basic = df_year[rc_basic_col].max()

        sc1 = ax1.scatter(df_year[lon_col], df_year[lat_col],
                          c=df_year[rc_basic_col], cmap=cmap,
                          transform=projection, s=20, alpha=0.7,
                          vmin=vmin_basic, vmax=vmax_basic, edgecolor='none')

        ax1.coastlines(resolution='50m')
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax1.set_title(f'Basic Radiative Cooling Potential ({rc_basic_col}) - {latest_year}', fontsize=12)
        cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.01, shrink=0.8)
        cbar1.set_label('RC Potential (W/m²)', fontsize=10)
        ax1.tick_params(labelsize=8)
        ax1.set_xlabel("Longitude", fontsize=10)
        ax1.set_ylabel("Latitude", fontsize=10)

        # Second subplot — Net RC
        ax2 = plt.subplot(1, 2, 2, projection=projection)
        vmin_net = max(0, df_year[rc_net_col].min())
        vmax_net = df_year[rc_net_col].max()

        sc2 = ax2.scatter(df_year[lon_col], df_year[lat_col],
                          c=df_year[rc_net_col], cmap=cmap,
                          transform=projection, s=20, alpha=0.7,
                          vmin=vmin_net, vmax=vmax_net, edgecolor='none')

        ax2.coastlines(resolution='50m')
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax2.set_title(f'Net Radiative Cooling Potential ({rc_net_col}) - {latest_year}', fontsize=12)
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.01, shrink=0.8)
        cbar2.set_label('RC Potential (W/m²)', fontsize=10)
        ax2.tick_params(labelsize=8)
        ax2.set_xlabel("Longitude", fontsize=10)
        ax2.set_ylabel("Latitude", fontsize=10)

        # Optional cluster overlay
        if cluster_col and cluster_col in df_year.columns:
            unique_clusters = sorted(df_year[cluster_col].unique())
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=f'C{i%10}', label=str(c), markersize=6)
                       for i, c in enumerate(unique_clusters)]
            ax2.legend(handles=handles, title='Cluster', loc='lower left')

        # Title and layout
        plt.suptitle(f'Annual Radiative Cooling Potential for Europe - {latest_year}', fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])

        # Save
        output_file = os.path.join(output_dir, f"yearly_rc_potential_{latest_year}.{OUTPUT_FORMAT}")
        fig.savefig(output_file, format=OUTPUT_FORMAT, dpi=300)
        plt.close(fig)

        print(f"✅ Saved yearly map to: {output_file}")

    except Exception as exc:
        print(f"❌ Error creating yearly maps: {exc}")

def create_seasonal_maps(seasonal_df, output_dir, boundaries, variable):
    """
    Create seasonal maps for a specific variable (e.g., P_rc_basic or P_rc_net).

    Parameters:
        seasonal_df (pandas.DataFrame): DataFrame with seasonal aggregated data
        output_dir (str): Directory to save output maps
        boundaries (tuple): (lon_min, lon_max, lat_min, lat_max) boundary coordinates
        variable (str): Variable to plot (e.g., 'P_rc_basic' or 'P_rc_net')
    """
    try:
        lon_min, lon_max, lat_min, lat_max = boundaries

        # Column access via config
        lon_col = get_column("lon")
        lat_col = get_column("lat")
        cluster_col = get_column("cluster_id", optional=True)
        season_col = get_column("season")
        year_col = get_column("year")

        # Select the most recent year
        latest_year = seasonal_df[year_col].max()
        df_year = seasonal_df[seasonal_df[year_col] == latest_year]

        # Validate variable presence
        if variable not in df_year.columns:
            print(f"⚠️ Variable '{variable}' not found in data. Available: {list(df_year.columns)}")
            return

        # Define seasonal order
        all_seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        available_seasons = df_year[season_col].unique()
        seasons = [s for s in all_seasons if s in available_seasons]

        if not seasons:
            print("⚠️ No season data available.")
            return

        # Create figure and layout
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(2, 2, figure=fig)
        projection = ccrs.PlateCarree()
        cmap = 'viridis'

        # Shared color scale
        vmin = max(0, df_year[variable].min())
        vmax = df_year[variable].max()

        for i, season in enumerate(seasons):
            if i >= 4:
                break

            df_season = df_year[df_year[season_col] == season]
            if df_season.empty:
                continue

            row, col = i // 2, i % 2
            ax = fig.add_subplot(gs[row, col], projection=projection)

            sc = ax.scatter(df_season[lon_col], df_season[lat_col],
                            c=df_season[variable], cmap=cmap,
                            transform=projection, s=20, alpha=0.7,
                            vmin=vmin, vmax=vmax, edgecolor='none')

            ax.coastlines(resolution='50m')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

            ax.set_title(f'{season} {latest_year}', fontsize=12)
            ax.tick_params(labelsize=8)
            ax.set_xlabel("Longitude", fontsize=10)
            ax.set_ylabel("Latitude", fontsize=10)

            if cluster_col and cluster_col in df_season.columns:
                unique_clusters = sorted(df_season[cluster_col].unique())
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=f'C{i%10}', label=str(c), markersize=6)
                           for i, c in enumerate(unique_clusters)]
                ax.legend(handles=handles, title='Cluster', loc='lower left')

        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label(f'{variable} (W/m²)', fontsize=12)

        # Overall title
        variable_label = "Basic" if "basic" in variable.lower() else "Net"
        plt.suptitle(f'Seasonal {variable_label} Radiative Cooling Potential for Europe - {latest_year}',
                     fontsize=16, y=0.98)

        # Final layout and save
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])
        variable_str = variable.lower().replace("p_rc_", "")
        output_file = os.path.join(output_dir,
                                   f"seasonal_rc_potential_{variable_str}_{latest_year}.{OUTPUT_FORMAT}")
        fig.savefig(output_file, format=OUTPUT_FORMAT, dpi=300)
        plt.close(fig)

        print(f"✅ Saved seasonal {variable} map to: {output_file}")

    except Exception as exc:
        print(f"❌ Error creating seasonal maps: {exc}")

def plot_kriged_rc_map(grid_lon, grid_lat, z_pred, save_path='rc_kriged_map.png'):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    cs = ax.pcolormesh(grid_lon, grid_lat, z_pred, cmap='YlOrRd',
                       shading='auto', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, orientation='vertical', pad=0.02)
    cbar.set_label("Kriged Annual RC Potential (kWh/m²·year)")

    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-30, 40, 35, 70], crs=ccrs.PlateCarree())

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Kriged Annual Radiative Cooling Potential", fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def run_visualization(yearly_df, seasonal_df, output_dir):
    """
    Run full visualization of RC maps for yearly and seasonal data.

    Parameters:
        yearly_df (pd.DataFrame): Aggregated yearly data
        seasonal_df (pd.DataFrame): Aggregated seasonal data
        output_dir (str): Path to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    boundaries = get_eu_boundaries(yearly_df)
    print(
        f"Map boundaries: Longitude [{boundaries[0]:.2f}, {boundaries[1]:.2f}], "
        f"Latitude [{boundaries[2]:.2f}, {boundaries[3]:.2f}]"
    )

    print("Generating yearly RC maps...")
    create_yearly_maps(yearly_df, output_dir, boundaries)

    print("Generating seasonal RC maps (basic)...")
    create_seasonal_maps(seasonal_df, output_dir, boundaries, variable=get_column("p_rc_basic"))

    print("Generating seasonal RC maps (net)...")
    create_seasonal_maps(seasonal_df, output_dir, boundaries, variable=get_column("p_rc_net"))

    print("✅ All RC maps generated.")

