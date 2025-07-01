import os
import logging
from clustering import main_matching_pipeline
import pandas as pd
from utils.feature_utils import save_config
from config import get_path


def multi_year_matching_pipeline(
    years,
    base_input_path,
    output_dir,
    borders_path,
    k_range=range(2, 10),
    db_url=None,
    db_table=None,
):
    """Run ``main_matching_pipeline`` across multiple yearly cluster files.

    Parameters
    ----------
    years : list[int]
        List of years to process.
    base_input_path : str
        Directory where ``clustered_dataset_<year>.csv`` files are stored.
    output_dir : str
        Directory to write matched datasets to.
    borders_path : str
        Shapefile used for mapping in ``main_matching_pipeline``.
    k_range : range, optional
        Range of ``k`` values to evaluate for clustering.
    """

    os.makedirs(output_dir, exist_ok=True)
    save_config(
        {
            "years": years,
            "base_input_path": base_input_path,
            "output_dir": output_dir,
            "borders_path": borders_path,
            "k_range": list(k_range),
            "db_url": db_url,
            "db_table": db_table,
        },
        os.path.join(output_dir, "logs"),
    )

    results = []

    for year in years:
        input_file = os.path.join(
            base_input_path,
            f"clustered_dataset_{year}.csv",
        )
        output_file = os.path.join(
            output_dir,
            f"matched_dataset_{year}.csv",
        )

        logging.info(f"\n=== Processing {year} ===")
        logging.info(f"Input: {input_file}")
        logging.info(f"Output: {output_file}")

        if db_url:
            from database_utils import read_table, write_dataframe
            df = read_table(db_table, db_url=db_url)
            df = main_matching_pipeline(
                clustered_data_path=input_file,
                shapefile_path=borders_path,
                output_file=output_file,
                k_range=k_range,
                db_url=db_url,
                db_table=db_table,
            )
            if df is not None:
                df["Year"] = year
                write_dataframe(df, db_table, db_url=db_url, if_exists="replace")
                results.append(df)
            continue

        if not os.path.exists(input_file):
            logging.error(f"❌ Input file not found for {year}")
            continue

        try:
            df = main_matching_pipeline(
                clustered_data_path=input_file,
                shapefile_path=borders_path,
                output_file=output_file,
                k_range=k_range,
            )

            if df is not None:
                df["Year"] = year
                results.append(df)
        except Exception as e:
            logging.error(f"❌ Error processing year {year}: {e}")

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined_file = os.path.join(
            output_dir,
            "matched_dataset_all_years.csv",
        )
        combined.to_csv(combined_file, index=False)
        logging.info(f"\n✅ Combined multi-year dataset saved to {combined_file}")
        return combined

    logging.warning("⚠️ No results generated.")
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-year matching")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    parser.add_argument("--base-input", default=os.path.join(get_path("results_path"), "clusters"))
    parser.add_argument("--output-dir", default=os.path.join(get_path("results_path"), "matching"))
    parser.add_argument("--borders-path", default=get_path("shapefile_path"))
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"))
    parser.add_argument("--db-table", default=os.getenv("PV_DB_TABLE", "pv_data"))
    args = parser.parse_args()

    multi_year_matching_pipeline(
        args.years,
        args.base_input,
        args.output_dir,
        args.borders_path,
        db_url=args.db_url,
        db_table=args.db_table,
    )
