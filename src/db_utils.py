import pandas as pd
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_db_from_csv(csv_path: str, db_path: str, table_name: str = "diabetes_raw"):
    """
    Loads data from a CSV file into a new SQLite database table.
    Overwrites the table if it exists.

    Args:
        csv_path (str): Path to the input CSV file.
        db_path (str): Path where the SQLite database file will be created/updated.
        table_name (str): Name of the table to create in the database.
    """
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        logging.info(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Ensure the directory for the database exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        logging.info(f"Connecting to SQLite database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        logging.info(f"Writing data to table '{table_name}'...")
        # Use 'replace' to overwrite the table if it already exists
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        logging.info(f"Data successfully written to {table_name} in {db_path}.")
        conn.commit()
        conn.close()
        logging.info("Database connection closed.")

    except Exception as e:
        logging.error(f"Error creating database from CSV: {e}")
        raise

if __name__ == "__main__":
    # Example usage: Run this script directly to create the DB
    # Assumes config file is accessible relative to this script's location if run directly,
    # or adjust paths as needed. For simplicity, using hardcoded paths for direct run:
    raw_csv = 'data/raw/diabetic_data.csv'
    database_path = 'data/database.db'
    create_db_from_csv(raw_csv, database_path)
    print(f"Database {database_path} created/updated.")