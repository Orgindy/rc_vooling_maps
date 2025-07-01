import os
import sys
import logging
from datetime import datetime
import pandas as pd
import argparse
from importlib import import_module
from packaging.requirements import Requirement
from pathlib import Path

try:
    from config import AppConfig, get_path, get_nc_dir
except Exception as exc:
    raise RuntimeError(f"Failed to load configuration: {exc}") from exc

from check_db_connection import main as check_db
from utils.resource_monitor import ResourceMonitor
from utils.file_operations import SafeFileOps
from utils.errors import ErrorAggregator, ProcessingError
from utils.feature_utils import save_config

from clustering import (
    prepare_clustered_dataset,
    main_matching_pipeline,
    plot_prediction_uncertainty_with_contours,
    compute_cluster_summary,
    compute_pv_potential_by_cluster_year,
    prepare_features_for_clustering,
)

from sklearn.model_selection import train_test_split
from train_models import train_all_models


def parse_args():
    parser = argparse.ArgumentParser(description="Run RC-PV pipeline")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "prep", "cluster"],
        help="Pipeline mode",
    )
    results_dir = get_path("results_path")
    default_input = os.path.join(results_dir, "clustered_dataset.csv")

    parser.add_argument(
        "--input-file",
        default=default_input,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--db-url", default=os.getenv("PV_DB_URL"), help="Optional database URL"
    )
    parser.add_argument(
        "--db-table",
        default=os.getenv("PV_DB_TABLE", "pv_data"),
        help="Table name if using DB",
    )
    parser.add_argument(
        "--k-range",
        default="2,8",
        help="Clustering k range, format: start,end",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="RC-PV Pipeline v1.0",
    )
    return parser.parse_args()


def validate_environment(args):
    """Validate DB URL if provided."""
    if args.db_url and not args.db_url.startswith("postgresql"):
        logging.error("Invalid PV_DB_URL. Expected PostgreSQL URL.")
        return False
    return True


def check_required_files(input_file):
    """Check if input file exists."""
    if not os.path.exists(input_file):
        logging.warning(f"Missing file: {input_file}")
        return False
    return True


def check_dependencies(requirements: Path):
    """Check all packages in requirements.txt are importable."""
    missing = []
    for line in requirements.read_text().splitlines():
        pkg = line.strip()
        if not pkg or pkg.startswith("#"):
            continue
        name = Requirement(pkg).name
        mod = name.replace("-", "_")
        try:
            import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        logging.error(f"Missing dependencies: {', '.join(missing)}")
        return False
    return True


def main_rc_pv_pipeline(input_path, db_url=None, db_table="pv_data", k_range=range(2, 8)):
    """Run full pipeline."""
    results_dir = get_path("results_path")
    data_dir = os.path.join(results_dir, "data")
    processed_path = os.path.join(data_dir, "clustered_dataset_enhanced.csv")
    output_path = os.path.join(results_dir, "matched_dataset.csv")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "maps"), exist_ok=True)

    resources = ResourceMonitor.check_system_resources()
    if not all(resources.values()):
        raise ProcessingError("Insufficient system resources", resources)

    if db_url:
        from database_utils import read_table, write_dataframe
        df_db = read_table(db_table, db_url=db_url)
        input_path = os.path.join(data_dir, "db_input.csv")
        df_db.to_csv(input_path, index=False)

    # === Step 1: Prepare dataset ===
    logging.info("🧼 Step 1: Data preparation")
    try:
        df_prepared = prepare_clustered_dataset(
            input_path=input_path, output_path=processed_path
        )
        if df_prepared is not None:
            logging.info(f"✅ Prepared: {len(df_prepared)} rows → {processed_path}")
        else:
            logging.warning("Using original input.")
            processed_path = input_path
    except Exception as e:
        logging.warning(f"Preparation failed: {e}")
        processed_path = input_path

    # === Step 1b: Train models ===
    if df_prepared is not None:
        logging.info("🤖 Step 1b: Train models")
        try:
            if "PV_Potential_physics" in df_prepared.columns:
                target = "PV_Potential_physics"
            elif "PV_Potential" in df_prepared.columns:
                target = "PV_Potential"
            else:
                target = None

            if target:
                X_scaled, _, _ = prepare_features_for_clustering(
                    df_prepared, use_predicted_pv=False
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, df_prepared[target], test_size=0.2, random_state=42
                )
                _, perf = train_all_models(X_train, X_test, y_train, y_test)
                perf_path = os.path.join(results_dir, "model_performance_summary.csv")
                pd.DataFrame(perf).T.to_csv(perf_path, index_label="Model")
                logging.info(f"✅ Models saved: {perf_path}")
            else:
                logging.warning("No target found for training.")
        except Exception:
            logging.exception("Model training failed.")

    # === Step 2: Clustering ===
    logging.info("🔗 Step 2: Clustering & matching")
    df_result = main_matching_pipeline(
        clustered_data_path=processed_path, output_file=output_path, k_range=k_range
    )
    if df_result is None:
        logging.error("❌ Clustering failed.")
        return None
    logging.info(f"✅ Clustering done: {output_path}")

    # === Step 3: Cluster summary ===
    try:
        compute_cluster_summary(df_result)
        logging.info("📊 Cluster summary saved.")
    except Exception:
        logging.exception("Cluster summary failed.")

    # === Step 4: PV potential summary ===
    try:
        compute_pv_potential_by_cluster_year(df_result)
        logging.info("📈 PV potential summary done.")
    except Exception:
        logging.exception("PV potential summary failed.")

    # === Step 5: Maps ===
    try:
        map_path = os.path.join(results_dir, "maps", "uncertainty_map.png")
        plot_prediction_uncertainty_with_contours(df_result, use_hatching=False, output_path=map_path)
        logging.info(f"🗺️ Map saved: {map_path}")
    except Exception:
        logging.exception("Map generation failed.")

    if db_url:
        from database_utils import write_dataframe
        write_dataframe(df_result, db_table, db_url=db_url, if_exists="replace")
        logging.info(f"✅ Results written to DB table: {db_table}")

    SafeFileOps.atomic_write(Path(output_path), df_result.to_csv(index=False))
    return df_result


def main():
    args = parse_args()
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    save_config(vars(args), logs_dir)

    if not validate_environment(args):
        return 1

    req_file = Path(__file__).parent / "requirements.txt"
    if not check_dependencies(req_file):
        logging.error("Dependency check failed.")
        return 1

    if not check_required_files(args.input_file):
        logging.error("Missing input file.")
        return 1

    k_start, k_end = map(int, args.k_range.split(","))
    k_range = range(k_start, k_end)

    results_dir = get_path("results_path")
    os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "maps"), exist_ok=True)

    logging.info(f"🚀 Running pipeline mode: {args.mode}")

    try:
        if args.mode == "full":
            result = main_rc_pv_pipeline(args.input_file, args.db_url, args.db_table, k_range)
        elif args.mode == "prep":
            result = prepare_clustered_dataset(
                input_path=args.input_file,
                output_path=os.path.join(results_dir, "data", "clustered_dataset_enhanced.csv")
            )
        elif args.mode == "cluster":
            result = main_matching_pipeline(
                clustered_data_path=args.input_file,
                output_file=os.path.join(results_dir, "matched_dataset.csv"),
                k_range=k_range
            )
        else:
            logging.error(f"Unknown mode: {args.mode}")
            return 1

        if result is not None:
            logging.info(f"✅ Pipeline done, rows: {len(result)}")
        else:
            logging.warning("Pipeline ended with no output.")

    except KeyboardInterrupt:
        logging.info("⏹️ Interrupted by user.")
    except Exception:
        logging.exception("Pipeline failed.")
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log"),
        ],
    )

    start = datetime.now()
    logging.info("=" * 50)
    logging.info("🌞 RC-PV PIPELINE STARTED")
    logging.info("=" * 50)

    try:
        exit_code = main()
    finally:
        logging.info(f"⏱️ Total runtime: {datetime.now() - start}")
        logging.info("🏁 PIPELINE FINISHED")
    sys.exit(exit_code)
