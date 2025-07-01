import xarray as xr
import os
import glob
import logging
import shutil
from tqdm import tqdm  # For progress bar
import sys
import cfgrib
from config import get_nc_dir, get_grib_dir, get_temp_dir
import eccodes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded paths - CHANGE THESE TO YOUR ACTUAL PATHS
GRIB_FOLDER = get_grib_dir()
NETCDF_OUTPUT_FOLDER = os.getenv("NETCDF_OUTPUT_FOLDER") or get_nc_dir()
MERGED_NETCDF_FILE = os.path.join(get_nc_dir(), os.getenv("MERGED_NETCDF_FILE", "era5_2023_merged.nc"))
TEMP_DIR = get_temp_dir()

# Configuration
MAX_FILES_IN_MEMORY = 4  # Maximum number of files to open simultaneously for merging

def validate_grib_file(grib_file):
    """
    Validates the GRIB file before conversion.

    Parameters:
    - grib_file (str): Path to the GRIB file.

    Returns:
    - bool: True if valid, False if corrupt or unreadable.
    """
    if not os.path.exists(grib_file) or os.path.getsize(grib_file) == 0:
        print(f"❌ File not found or empty: {grib_file}")
        return False

    try:
        with open(grib_file, 'rb') as f:
            handle = eccodes.codes_grib_new_from_file(f)
            if handle is None:
                print(f"❌ Not a valid GRIB file: {grib_file}")
                return False
            eccodes.codes_release(handle)
        return True
    except Exception as e:
        print(f"❌ GRIB validation failed for {grib_file}: {e}")
        return False

def validate_metadata(nc_file):
    """
    Validates the metadata of a NetCDF file to ensure consistency.
    
    Parameters:
    - nc_file (str): Path to the NetCDF file to validate.
    
    Returns:
    - bool: True if the file passes validation, False otherwise.
    """
    try:
        # Open the NetCDF file within a context manager to ensure it closes
        with xr.open_dataset(nc_file) as ds:
            # Check for critical dimensions
            required_dims = ["time", "latitude", "longitude"]
            missing_dims = [dim for dim in required_dims if dim not in ds.dims]

            if missing_dims:
                print(f"❌ Missing critical dimensions in {nc_file}: {missing_dims}")
                return False

            # Check for required global attributes
            required_attrs = ["Conventions", "title", "institution", "source"]
            missing_attrs = [attr for attr in required_attrs if attr not in ds.attrs]

            if missing_attrs:
                print(f"⚠️ Missing global attributes in {nc_file}: {missing_attrs}")
                return False

            # Check for variable consistency
            if not all(var in ds.data_vars for var in ["t2m", "sp", "u10", "v10"]):
                print(f"⚠️ Missing expected data variables in {nc_file}")
                return False

            print(f"✅ {nc_file} passed metadata validation.")
            return True
    
    except Exception as e:
        print(f"❌ Failed to validate {nc_file}: {e}")
        return False

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def grib_to_netcdf(input_file, output_folder):
    """
    Convert a GRIB file to NetCDF format.
    
    Parameters:
    -----------
    input_file : str
        Path to the input GRIB file
    output_folder : str
        Path to the folder where NetCDF files will be saved
    
    Returns:
    --------
    str or None
        Path to the output NetCDF file if successful, None otherwise
    """
    # Create output filename based on input filename
    base_filename = os.path.basename(input_file)
    base_name = os.path.splitext(base_filename)[0]
    output_file = os.path.join(output_folder, f"{base_name}.nc")
    
    try:
        ds = xr.open_dataset(
            input_file,
            engine='cfgrib',
            backend_kwargs={'errors': 'ignore', 'filter_by_keys': {}}
        )
    
        try:
            ds.to_netcdf(
                output_file,
                encoding={var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
            )
        except Exception as e:
            logger.error(f"Failed to write NetCDF for {base_filename}: {e}")
            ds.close()
            return None
        
        # Close the dataset to free memory
        ds.close()
        
        logger.info(f"Converted {base_filename} to NetCDF successfully")
        return output_file
    
    except Exception as e:
        logger.error(f"Error opening GRIB file {base_filename}: {e}")
        return None

def identify_merge_dimension(netcdf_files, sample_size=3):
    """
    Identify the best dimension to merge along by examining a sample of files.
    
    Parameters:
    -----------
    netcdf_files : list
        List of NetCDF file paths
    sample_size : int
        Number of files to examine for dimension detection
    
    Returns:
    --------
    str or None
        Name of the dimension to merge along, or None if no suitable dimension is found
    """
    if len(netcdf_files) <= 1:
        return None
        
    # Take a sample of files to examine
    sample_files = netcdf_files[:min(sample_size, len(netcdf_files))]
    
    try:
        # Open sample datasets
        sample_datasets = [xr.open_dataset(file) for file in sample_files]
        
        # Common dimensions to check in order of preference
        dimensions_to_check = ['time', 'step', 'valid_time', 'forecast_time', 'realization']
        
        # Check each dimension
        for dim in dimensions_to_check:
            # Check if all datasets have this dimension
            if all(dim in ds.dims for ds in sample_datasets):
                # Check if values are different across datasets (makes sense to concatenate)
                
                # Get values from each dataset carefully
                dim_values = []
                for ds in sample_datasets:
                    # Handle different data formats safely
                    values = ds[dim].values
                    # Just store a hash or representative value to compare
                    if hasattr(values, 'size') and values.size > 0:
                        # For numpy arrays
                        dim_values.append(hash(values.tobytes()))
                    elif hasattr(values, '__iter__') and len(values) > 0:
                        # For regular iterables
                        dim_values.append(hash(str(values[0])))
                    else:
                        # For single values
                        dim_values.append(hash(str(values)))
                
                # Check if we have at least 2 different sets of values
                if len(set(dim_values)) > 1:
                    logger.info(f"Identified '{dim}' as suitable merge dimension")
                    
                    # Close all datasets
                    for ds in sample_datasets:
                        ds.close()
                        
                    return dim
        
        # If no suitable dimension found, use coordinate-based merge
        logger.info("No suitable dimension found for concatenation, will use coordinate-based merge")
        
        # Close all datasets
        for ds in sample_datasets:
            ds.close()
            
        return None
        
    except Exception as e:
        logger.error(f"Error identifying merge dimension: {e}")
        # Ensure datasets are closed
        try:
            for ds in sample_datasets:
                ds.close()
        except Exception as e:
            pass
        return None

def merge_netcdf_files_chunked(netcdf_files, output_file, merge_dim=None, chunk_size=None):
    """
    Memory-efficient merging of NetCDF files by processing in chunks.
    
    Parameters:
    -----------
    netcdf_files : list
        List of NetCDF file paths to merge
    output_file : str
        Path to the output merged NetCDF file
    merge_dim : str or None
        Dimension to merge along, or None for automatic detection
    chunk_size : int or None
        Number of files to process at once, defaults to MAX_FILES_IN_MEMORY
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if not netcdf_files:
        logger.error("No NetCDF files to merge")
        return False
        
    if len(netcdf_files) == 1:
        logger.info("Only one file to process, copying instead of merging")
        shutil.copy2(netcdf_files[0], output_file)
        return True
    
    # Use default chunk size if not specified
    if chunk_size is None:
        chunk_size = MAX_FILES_IN_MEMORY
    
    # Create temp directory
    ensure_dir_exists(TEMP_DIR)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    ensure_dir_exists(output_dir)
    
    try:
        # Identify merge dimension if not provided
        if merge_dim is None:
            merge_dim = identify_merge_dimension(netcdf_files)
        
        # Divide files into manageable chunks
        chunks = [netcdf_files[i:i+chunk_size] for i in range(0, len(netcdf_files), chunk_size)]
        logger.info(f"Processing {len(netcdf_files)} files in {len(chunks)} chunks")
        
        # Process each chunk
        temp_files = []
        for i, chunk in enumerate(tqdm(chunks, desc="Merging chunks")):
            temp_file = os.path.join(TEMP_DIR, f"temp_merge_{i}.nc")
            
            # Open datasets in this chunk
            datasets = [xr.open_dataset(file) for file in chunk]
            
            # Merge datasets
            if merge_dim and all(merge_dim in ds.dims for ds in datasets):
                # Merge along specified dimension
                merged_ds = xr.concat(datasets, dim=merge_dim)
                logger.debug(f"Merged chunk {i+1}/{len(chunks)} along {merge_dim} dimension")
            else:
                # Fall back to coordinate-based merge
                merged_ds = xr.merge(datasets)
                logger.debug(f"Merged chunk {i+1}/{len(chunks)} using coordinate-based merge")
            
            # Save merged chunk
            merged_ds.to_netcdf(
                temp_file,
                encoding={var: {'zlib': True, 'complevel': 5} for var in merged_ds.data_vars}
            )
            
            # Close all datasets
            for ds in datasets:
                ds.close()
            
            # Close merged dataset
            merged_ds.close()
            
            temp_files.append(temp_file)
        
        # Now merge the temporary files
        if len(temp_files) == 1:
            # Only one temp file, just rename it
            shutil.move(temp_files[0], output_file)
        else:
            # Merge temp files using the same method
            logger.info(f"Merging {len(temp_files)} intermediate files")
            
            # Open all temp datasets
            temp_datasets = [xr.open_dataset(file) for file in temp_files]
            
            # Merge temp datasets
            if merge_dim and all(merge_dim in ds.dims for ds in temp_datasets):
                final_merged = xr.concat(temp_datasets, dim=merge_dim)
            else:
                final_merged = xr.merge(temp_datasets)
            
            # Save final merged file
            final_merged.to_netcdf(
                output_file,
                encoding={var: {'zlib': True, 'complevel': 5} for var in final_merged.data_vars}
            )
            
            # Close all temp datasets
            for ds in temp_datasets:
                ds.close()
                
            # Close final merged dataset
            final_merged.close()
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        logger.info(f"Successfully merged all files into {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        logger.exception("Detailed traceback:")
        return False
    finally:
        # Make sure we clean up temp files even if there was an error
        for temp_file in glob.glob(os.path.join(TEMP_DIR, "temp_merge_*.nc")):
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    pass

def main():
    # Ensure required directories exist
    ensure_dir_exists(NETCDF_OUTPUT_FOLDER)
    ensure_dir_exists(TEMP_DIR)

    # Step 1: Convert GRIB → NetCDF with GRIB validation
    grib_files = sorted(glob.glob(os.path.join(GRIB_FOLDER, "*.grib")) +
                        glob.glob(os.path.join(GRIB_FOLDER, "*.grib2")))
    netcdf_files = []

    for grib_file in tqdm(grib_files, desc="Validating and converting GRIB files"):
        if validate_grib_file(grib_file):
            nc_file = grib_to_netcdf(grib_file, NETCDF_OUTPUT_FOLDER, TEMP_DIR)
            if nc_file:
                netcdf_files.append(nc_file)
        else:
            print(f"⚠️ Skipping invalid GRIB: {grib_file}")

    if not netcdf_files:
        print("❌ No valid GRIB files were converted. Exiting.")
        return

    # (Optional) Step 2: Validate NetCDF files
    valid_netcdf_files = []
    for nc in netcdf_files:
        if validate_metadata(nc):
            valid_netcdf_files.append(nc)
        else:
            print(f"⚠️ Skipping invalid NetCDF: {nc}")

    if not valid_netcdf_files:
        print("❌ No valid NetCDF files to merge. Exiting.")
        return

    # Step 3: Merge NetCDF files
    merge_success = merge_netcdf_files_chunked(valid_netcdf_files, MERGED_NETCDF_FILE, merge_dim="time")

    if not merge_success:
        print("❌ Merging failed.")
        return

    print("🎉 All tasks completed successfully.")