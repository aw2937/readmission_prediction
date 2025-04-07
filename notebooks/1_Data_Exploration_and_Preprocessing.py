# -*- coding: utf-8 -*-
"""
Notebook for Exploring, Cleaning, and Preprocessing the Diabetes Dataset.
"""

# %% [markdown]
# # 1. Data Exploration and Preprocessing
#
# **Goal:** Load the raw diabetes dataset, explore its characteristics, clean it, perform feature engineering/selection, preprocess it for modeling, and save the final train, validation, and test sets.
#
# **Skills Demonstrated:**
# - Python, Pandas, Numpy for data manipulation
# - Matplotlib, Seaborn for Exploratory Data Analysis (EDA)
# - Scikit-learn for preprocessing (Splitting, Scaling, Encoding)
# - Handling missing values
# - Feature engineering (basic)
# - Understanding data distributions and relationships
# - Structured approach using functions from `src` modules

# %%
# Import necessary libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import sys

# Add src directory to Python path to import custom modules
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src import data_loader, preprocess

# Configure plotting style
sns.set(style="whitegrid")
%matplotlib inline

# %% [markdown]
# ## 1.1 Load Configuration and Raw Data
#
# Load the project configuration and the raw dataset using our `data_loader` module. We'll also optionally demonstrate loading from the SQLite DB if `db_utils.py` was run.

# %%
# Load config
config_path = '../config/model_config.yaml' # Path relative to notebook location
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

paths = config['paths']
cfg_prep = config['preprocessing']

# %%
# Load raw data from CSV
try:
    df_raw = data_loader.load_raw_data(f"../{paths['raw_data']}") # Adjust path relative to notebook
    print("Raw data loaded from CSV:")
    print(df_raw.head())
    print(df_raw.info())
except FileNotFoundError as e:
    print(e)
    print("Please ensure 'diabetic_data.csv' is in the 'data/raw/' directory.")
    # Stop execution if file not found
    # raise

# %%
# (Optional) Load raw data from SQLite DB - Uncomment if you ran db_utils.py
# db_path_relative = f"../{paths['db_path']}"
# if os.path.exists(db_path_relative):
#     print("\nLoading raw data from SQLite DB...")
#     df_raw_db = data_loader.load_data_from_db(db_path_relative)
#     if df_raw_db is not None:
#         print(df_raw_db.head())
#     else:
#         print("Could not load data from DB.")
# else:
#     print("\nSQLite DB not found. Run src/db_utils.py to create it.")


# %% [markdown]
# ## 1.2 Initial Data Exploration (EDA)
# Perform basic EDA on the raw data.
# - Check dimensions
# - Look at data types
# - Examine missing values (represented as '?')
# - Analyze the target variable distribution

# %%
print(f"Dataset dimensions: {df_raw.shape}")

# %%
# Check for '?' representing missing values
print("\nColumns with '?' values:")
missing_q = (df_raw == '?').sum()
print(missing_q[missing_q > 0])

# %%
# Analyze target variable 'readmitted'
print("\nTarget variable distribution ('readmitted'):")
print(df_raw['readmitted'].value_counts())
print("\nNormalized distribution:")
print(df_raw['readmitted'].value_counts(normalize=True))

plt.figure(figsize=(8, 5))
sns.countplot(data=df_raw, x='readmitted')
plt.title('Distribution of Readmission Status')
plt.show()

# %% [markdown]
# **Observations:**
# - Several columns (`weight`, `payer_code`, `medical_specialty`) have a high number of missing values ('?'). These are candidates for dropping.
# - Other columns also have missing values that need handling.
# - The target variable shows imbalance: 'NO' readmission is the most frequent category. We are interested in predicting `<30` days readmission, making this a binary classification problem after mapping.

# %% [markdown]
# ## 1.3 Data Cleaning and Target Mapping
# Apply the cleaning steps defined in `src/preprocess.py`.
# - Replace '?' with NaN.
# - Drop specified columns.
# - Drop rows with missing values in key columns (simplification for demo).
# - Handle duplicates (keeping first encounter per patient).
# - Map categorical 'age' to numerical.
# - Map the target variable `readmitted` to binary (1 for `<30`, 0 otherwise).

# %%
# Apply cleaning function
df_cleaned = preprocess.clean_data(df_raw.copy(), cfg_prep['drop_cols']) # Use copy to avoid modifying df_raw

# %%
# Apply target mapping function
df_mapped = preprocess.map_target(df_cleaned, cfg_prep['target_col'], cfg_prep['target_map'])

print("\nTarget variable distribution after mapping:")
print(df_mapped[cfg_prep['target_col']].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(data=df_mapped, x=cfg_prep['target_col'])
plt.title('Distribution of Binary Target (1 = Readmitted <30d)')
plt.xticks([0, 1], ['No / >30d', '<30d'])
plt.show()

# %% [markdown]
# **Observation:** The target variable remains imbalanced after mapping to binary, with the positive class (`<30` days readmission) being the minority. This needs to be addressed during modeling (e.g., class weights, appropriate metrics).

# %% [markdown]
# ## 1.4 Feature Identification and Splitting
# Identify numerical and categorical features and split the data into training, validation, and test sets *before* applying scaling and encoding.

# %%
# Identify feature types (as done in preprocess_pipeline)
potential_numeric = cfg_prep['numeric_cols']
all_cols = df_mapped.drop(columns=[cfg_prep['target_col']]).columns
numeric_features = [col for col in potential_numeric if col in all_cols]
categorical_features = [col for col in all_cols if col not in numeric_features]

print(f"\nIdentified {len(numeric_features)} numeric features.")
# print(numeric_features)
print(f"Identified {len(categorical_features)} categorical features.")
# print(categorical_features)

# %%
# Split data using the function from preprocess.py
train_df_unprocessed, val_df_unprocessed, test_df_unprocessed = preprocess.split_data(
    df_mapped,
    cfg_prep['target_col'],
    cfg_prep['test_size'],
    cfg_prep['validation_size'],
    cfg_prep['random_state']
)

print("\nData split summary:")
print(f"Train set shape (unprocessed): {train_df_unprocessed.shape}")
print(f"Validation set shape (unprocessed): {val_df_unprocessed.shape}")
print(f"Test set shape (unprocessed): {test_df_unprocessed.shape}")

# %% [markdown]
# ## 1.5 Preprocessing (Scaling and Encoding) and Saving
# - Create the preprocessor (StandardScaler for numeric, OneHotEncoder for categorical).
# - **Fit** the preprocessor **only** on the **training data**.
# - **Transform** all three sets (train, validation, test) using the **fitted** preprocessor.
# - Save the processed datasets.

# %%
# Create the preprocessor
preprocessor = preprocess.create_preprocessor(numeric_features, categorical_features)

# Fit on training data
X_train_unprocessed = train_df_unprocessed.drop(columns=[cfg_prep['target_col']])
print("\nFitting preprocessor on training data...")
preprocessor.fit(X_train_unprocessed)
print("Preprocessor fitted.")

# %%
# Define the apply_preprocessing helper function (copied from preprocess.py for notebook use)
def apply_preprocessing_notebook(df_in: pd.DataFrame, target_col: str, prep) -> pd.DataFrame:
    X = df_in.drop(columns=[target_col])
    y = df_in[target_col]
    print(f"Applying preprocessor to data with shape: {X.shape}")
    X_processed = prep.transform(X)
    # Get feature names after transformation
    try:
        # For sklearn >= 1.0
        feature_names = prep.get_feature_names_out()
    except AttributeError:
         # Fallback for older sklearn or different ColumnTransformer structure
         # This part might need adjustment based on the exact sklearn version and preprocessor steps
        feature_names = []
        for name, trans, cols in prep.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                 # Handle transformers like OneHotEncoder
                f_names = trans.get_feature_names_out(cols)
                feature_names.extend(f_names)
            elif name != 'remainder':
                 # Handle transformers like StandardScaler
                 feature_names.extend(cols)
            elif trans == 'passthrough':
                 # Handle remainder='passthrough'
                 rem_cols = [c for c in X.columns if c not in prep._feature_names_in] # Access internal attr, maybe fragile
                 feature_names.extend(rem_cols)


    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    print(f"Processed data shape: {X_processed_df.shape}")
    # Ensure y index matches X_processed_df before concatenation
    y = y.loc[X_processed_df.index] # Align index
    return pd.concat([X_processed_df, y], axis=1)


# %%
# Apply preprocessing to all splits
print("\nApplying preprocessing to training set...")
train_processed_df = apply_preprocessing_notebook(train_df_unprocessed, cfg_prep['target_col'], preprocessor)

print("\nApplying preprocessing to validation set...")
val_processed_df = apply_preprocessing_notebook(val_df_unprocessed, cfg_prep['target_col'], preprocessor)

print("\nApplying preprocessing to test set...")
test_processed_df = apply_preprocessing_notebook(test_df_unprocessed, cfg_prep['target_col'], preprocessor)

# %%
# Display head of processed training data
print("\nHead of processed training data:")
print(train_processed_df.head())
print(f"\nNumber of features after preprocessing: {train_processed_df.shape[1] - 1}") # -1 for target col

# %% [markdown]
# **Observation:** One-Hot Encoding significantly increased the number of features, especially due to diagnosis codes and medications. This high dimensionality might pose challenges for some models but is handled by Neural Networks. Feature selection or embedding layers could be considered for optimization.

# %%
# Save processed data
processed_dir = f"../{paths['processed_data_dir']}"
os.makedirs(processed_dir, exist_ok=True)

train_save_path = f"../{paths['train_data']}"
val_save_path = f"../{paths['val_data']}"
test_save_path = f"../{paths['test_data']}"

data_loader.save_processed_data(train_processed_df, train_save_path)
data_loader.save_processed_data(val_processed_df, val_save_path)
data_loader.save_processed_data(test_processed_df, test_save_path)

print(f"\nProcessed data saved to {processed_dir}")

# %% [markdown]
# ## 1.6 Conclusion
#
# The data has been successfully explored, cleaned, preprocessed, and split into training, validation, and test sets. The resulting CSV files in `data/processed/` are ready to be used for model training and evaluation. The preprocessing steps (scaling, encoding) were fitted only on the training data to prevent data leakage.