import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import yaml
import os
import argparse
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules from src
from . import data_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_confusion_matrix(y_true, y_pred, save_path: str, threshold=0.5):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred >= threshold)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No Readmit', 'Predicted Readmit (<30d)'], yticklabels=['Actual No Readmit', 'Actual Readmit (<30d)'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f"Confusion matrix plot saved to: {save_path}")
    plt.close()

def evaluate_model(config_path: str):
    """Loads the test data and the saved model, evaluates it, and saves metrics/plots."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    target_col = config['preprocessing']['target_col']

    # Load test data
    logging.info("Loading test data...")
    try:
        test_df = data_loader.load_processed_data(paths['test_data'])
    except FileNotFoundError:
        logging.error("Test data file not found. Please run the preprocessing notebook first.")
        return

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    logging.info(f"Test data shape: {X_test.shape}")

    # Load the saved model
    model_path = paths['model_save_path']
    if not os.path.exists(model_path):
        logging.error(f"Saved model not found at: {model_path}. Please run the training script first.")
        return

    logging.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    logging.info("Model loaded successfully.")

    # Make predictions
    logging.info("Making predictions on the test set...")
    y_pred_proba = model.predict(X_test).flatten() # Flatten for binary classification output
    y_pred_class = (y_pred_proba >= 0.5).astype(int) # Apply threshold for class labels

    # Evaluate
    logging.info("Evaluating model performance...")
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred_class, target_names=['No Readmit', 'Readmit (<30d)'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred_class)

    logging.info("\n--- Test Set Evaluation ---")
    logging.info(f"ROC AUC Score: {auc:.4f}")
    logging.info("Classification Report:")
    # Pretty print the dict report
    print(classification_report(y_test, y_pred_class, target_names=['No Readmit', 'Readmit (<30d)']))
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")
    logging.info("---------------------------\n")


    # Save metrics
    metrics = {
        'roc_auc': auc,
        'classification_report': report,
        'confusion_matrix': cm.tolist() # Convert numpy array to list for JSON serialization
    }

    metrics_path = paths['metrics_save_path']
    # Ensure directory exists
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Evaluation metrics saved to: {metrics_path}")

    # Save confusion matrix plot
    cm_plot_path = os.path.join(paths['plot_save_dir'], "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred_proba, cm_plot_path) # Use probabilities for plotting potentially

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained patient readmission prediction model.")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()
    evaluate_model(args.config)
