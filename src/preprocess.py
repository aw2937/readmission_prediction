import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Tuple, Dict, Any, List
from . import data_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """Basic cleaning: Replace '?' with NaN, drop specified columns and rows with missing values in key columns."""
    logging.info("Starting data cleaning...")
    df = df.replace('?', np.nan)
    logging.info(f"Replaced '?' with NaN. Shape: {df.shape}")

    # Drop columns with too many missing values or irrelevant ones
    df = df.drop(columns=drop_cols, errors='ignore')
    logging.info(f"Dropped specified columns. Shape: {df.shape}")

    # Drop rows where target or key predictors are missing (example: race, gender, diag_1)
    # In a real scenario, more sophisticated imputation might be needed
    initial_rows = len(df)
    df = df.dropna(subset=['race', 'gender', 'diag_1', 'diag_2', 'diag_3', 'readmitted'])
    rows_dropped = initial_rows - len(df)
    logging.info(f"Dropped {rows_dropped} rows with missing values in key columns. Shape: {df.shape}")

    # Drop duplicate patient encounters based on patient_nbr if it wasn't dropped
    # Keeping only the first encounter per patient for simplicity here
    if 'patient_nbr' in df.columns:
         initial_rows = len(df)
         df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
         rows_dropped = initial_rows - len(df)
         logging.info(f"Dropped {rows_dropped} duplicate patient encounters. Shape: {df.shape}")


    # Convert age category to numerical representation (mid-point or ordered integer)
    if 'age' in df.columns:
        age_map = {f'[{10*i}-{10*(i+1)})': 10*i + 5 for i in range(10)}
        df['age'] = df['age'].map(age_map)
        logging.info("Mapped 'age' column to numerical values.")

    # Optional: Simplify medication columns (Example: map 'No', 'Steady', 'Up', 'Down' to 0, 1, 2, 3 or similar)
    # This could be complex, skipping detailed mapping for brevity in this demo.

    logging.info("Data cleaning finished.")
    return df

def map_target(df: pd.DataFrame, target_col: str, target_map: Dict[str, int]) -> pd.DataFrame:
    """Maps the target variable to numerical values."""
    logging.info(f"Mapping target variable '{target_col}'...")
    df[target_col] = df[target_col].map(target_map)
    # Drop rows where target mapping resulted in NaN (if any unexpected values existed)
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
    logging.info(f"Target variable mapped. Value counts:\n{df[target_col].value_counts(normalize=True)}")
    return df


def create_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Creates a Sklearn ColumnTransformer for preprocessing."""
    logging.info("Creating preprocessor pipeline...")

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # dense output for TF
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like target) if not dropped
        # Or use remainder='drop' if only features should be passed through
    )
    logging.info("Preprocessor created.")
    return preprocessor

def split_data(df: pd.DataFrame, target_col: str, test_size: float, validation_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data into training, validation, and test sets."""
    logging.info("Splitting data into train, validation, and test sets...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split into train+validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Adjust validation size relative to the train_val set
    val_size_adjusted = validation_size / (1 - test_size)

    # Split train+validation into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=y_train_val
    )

    logging.info(f"Data split complete:")
    logging.info(f"Train set shape: {X_train.shape}, Target distribution:\n{y_train.value_counts(normalize=True)}")
    logging.info(f"Validation set shape: {X_val.shape}, Target distribution:\n{y_val.value_counts(normalize=True)}")
    logging.info(f"Test set shape: {X_test.shape}, Target distribution:\n{y_test.value_counts(normalize=True)}")

    # Combine features and target back for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, val_df, test_df

def preprocess_pipeline(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs the full preprocessing pipeline."""
    paths = config['paths']
    cfg = config['preprocessing']

    # Load data (using raw CSV loader here)
    df = data_loader.load_raw_data(paths['raw_data'])

    # Clean data
    df = clean_data(df, cfg['drop_cols'])

    # Map target variable
    df = map_target(df, cfg['target_col'], cfg['target_map'])

    # Identify feature types dynamically after cleaning
    potential_numeric = cfg['numeric_cols']
    all_cols = df.drop(columns=[cfg['target_col']]).columns
    numeric_features = [col for col in potential_numeric if col in all_cols]
    categorical_features = [col for col in all_cols if col not in numeric_features]
    logging.info(f"Identified {len(numeric_features)} numeric features.")
    logging.info(f"Identified {len(categorical_features)} categorical features.")


    # Split data BEFORE applying transformations that require fitting (like OHE, Scaler)
    train_df, val_df, test_df = split_data(
        df,
        cfg['target_col'],
        cfg['test_size'],
        cfg['validation_size'],
        cfg['random_state']
    )

    # Create and fit preprocessor ONLY on training data
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    logging.info("Fitting preprocessor on training data...")

    X_train = train_df.drop(columns=[cfg['target_col']])
    y_train = train_df[cfg['target_col']]
    preprocessor.fit(X_train) # Fit the scaler and encoder

    # Function to apply the fitted preprocessor and reconstruct DataFrame
    def apply_preprocessing(df_in: pd.DataFrame, target_col: str, prep: ColumnTransformer) -> pd.DataFrame:
        X = df_in.drop(columns=[target_col])
        y = df_in[target_col]
        logging.info(f"Applying preprocessor to data with shape: {X.shape}")
        X_processed = prep.transform(X)
        # Get feature names after transformation (important!)
        feature_names = prep.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        logging.info(f"Processed data shape: {X_processed_df.shape}")
        return pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1) # Reset index of y for safe concat

    # Apply preprocessing to all splits
    train_processed_df = apply_preprocessing(train_df, cfg['target_col'], preprocessor)
    val_processed_df = apply_preprocessing(val_df, cfg['target_col'], preprocessor)
    test_processed_df = apply_preprocessing(test_df, cfg['target_col'], preprocessor)

    # Save processed data
    data_loader.save_processed_data(train_processed_df, paths['train_data'])
    data_loader.save_processed_data(val_processed_df, paths['val_data'])
    data_loader.save_processed_data(test_processed_df, paths['test_data'])

    return train_processed_df, val_processed_df, test_processed_df
