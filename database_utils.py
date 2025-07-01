from pathlib import Path
import importlib.util
import sys

# Load legacy helper if present
legacy_path = Path(__file__).with_name("database_utils(dont need).py")
if legacy_path.exists():
    spec = importlib.util.spec_from_file_location("_db_legacy", legacy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Re-export public functions
    read_table = getattr(module, "read_table")
    write_dataframe = getattr(module, "write_dataframe")
else:
    raise ImportError("database utilities not available")
