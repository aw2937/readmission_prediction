import pandas as pd
import sqlite3
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_data(path: str) -> pd.DataFrame:
    """Loads raw data from a CSV file."""
    logging.info(f"Loading raw data from: {path}")
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path)
        logging.info(f"Raw data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading raw data from {path}: {e}")
        raise

def load_data_from_db(db_path: str, table_name: str = "diabetes_raw") -> Optional[pd.DataFrame]:
    """Loads data from an SQLite database table."""
    logging.info(f"Loading data from table '{table_name}' in database: {db_path}")
    if not os.path.exists(db_path):
        logging.warning(f"Database file not found: {db_path}. Run db_utils.py first.")
        return None
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        logging.info(f"Data loaded successfully from DB. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from database {db_path}: {e}")
        return None # Return None or raise specific exception

def load_processed_data(path: str) -> pd.DataFrame:
    """Loads processed data from a CSV file."""
    logging.info(f"Loading processed data from: {path}")
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path)
        logging.info(f"Processed data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading processed data from {path}: {e}")
        raise

def save_processed_data(df: pd.DataFrame, path: str):
    """Saves processed data to a CSV file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Data saved successfully to: {path}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise