import argparse
import logging
from database_utils import get_engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Check database connection")
parser.add_argument("--db-url", required=True)
args = parser.parse_args()

engine = get_engine(args.db_url)
if engine is not None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Successfully connected")
else:
    print("Failed to connect")
