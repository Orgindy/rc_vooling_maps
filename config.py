from dataclasses import dataclass
from pathlib import Path
import os
import logging
import yaml
from typing import Dict, Any, Optional

# === Locate config.yaml ===
CONFIG_FILE = Path(__file__).with_name("config.yaml")

# === Load config ===
def load_config(path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration values from a YAML file."""
    if not path.exists():
        logging.warning("Config file %s not found.", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logging.warning("Error reading config file: %s", e)
        return {}

# === Validate required keys ===
REQUIRED_KEYS = ["DATA_FOLDER", "MODEL_PATH", "DB_PATH"]

def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required keys and create default output dirs."""
    for key in REQUIRED_KEYS:
        value = cfg.get(key)
        if value is None:
            logging.warning("Missing required config key: %s", key)
            raise KeyError(f"Missing config key: {key}")
        if ("PATH" in key or "DIR" in key) and isinstance(value, str):
            if not os.path.exists(value):
                logging.warning("Configured path for %s does not exist: %s", key, value)
    output_dir = cfg.get("OUTPUT_DIR", "./outputs")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cfg.setdefault("OUTPUT_DIR", output_dir)
    return cfg

CONFIG = validate_config(load_config())

# === Path getter ===
def get_path(key: str, default: Optional[str] = None) -> Path:
    """Return a configured path as a ``Path`` object, validating existence."""
    value = CONFIG.get(key, default)
    if value is None:
        logging.warning(
            "Configuration key %s not found and no default provided", key
        )
        return Path()

    path = Path(value)
    if ("PATH" in key.upper() or "DIR" in key.upper()) and not path.exists():
        raise FileNotFoundError(
            f"Configured path for {key} does not exist: {path}"
        )

    return path

# === NetCDF directory helper ===
def get_nc_dir() -> str:
    """Return directory containing NetCDF files."""
    env_dir = os.getenv("NC_DATA_DIR")
    if env_dir:
        return env_dir
    return get_path("era5_path", "netcdf_files")

# === App-wide resource config ===
@dataclass
class AppConfig:
    """Application-wide resource limits."""
    memory_limit: float = 75.0
    disk_space_min_gb: float = 10.0
    max_file_size: int = 100 * 1024 * 1024
    cpu_max_percent: float = 90.0

    resource_check_interval: int = 60
    enable_file_locking: bool = True
    temp_cleanup_interval: int = 3600

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load resource limits from environment vars."""
        return cls(
            memory_limit=float(os.getenv("MAX_MEMORY_PERCENT", cls.memory_limit)),
            disk_space_min_gb=float(os.getenv("DISK_SPACE_MIN_GB", cls.disk_space_min_gb)),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", cls.max_file_size)),
            cpu_max_percent=float(os.getenv("CPU_MAX_PERCENT", cls.cpu_max_percent)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'AppConfig':
        """Load AppConfig from YAML."""
        if not path.exists():
            return cls()
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self) -> Optional[str]:
        """Validate resource values."""
        if not 0 < self.memory_limit <= 100:
            return "memory_limit must be between 0 and 100"
        if self.disk_space_min_gb <= 0:
            return "disk_space_min_gb must be positive"
        if self.max_file_size <= 0:
            return "max_file_size must be positive"
        if not 0 < self.cpu_max_percent <= 100:
            return "cpu_max_percent must be between 0 and 100"
        return None

# === Unified TrainingConfig ===
@dataclass
class TrainingConfig:
    """Paths for training datasets and flags."""
    train_features: str
    test_features: str
    train_target_pv: str
    test_target_pv: str
    train_target_cooling: str
    test_target_cooling: str
    train_target_net: str
    test_target_net: str
    use_kg_features: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load TrainingConfig from YAML."""
        if not path.exists():
            raise FileNotFoundError(f"Training config YAML not found: {path}")
        with path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(**cfg)
