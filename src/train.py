import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
import yaml
import os
import argparse
import logging
import matplotlib.pyplot as plt

# Import modules from src
# Import modules from src
from . import data_loader
from . import model as model_builder # Alias to avoid conflict with keras.Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_training_history(history, save_path: str):
    """Plots accuracy and loss curves for training and validation sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)


    # Plot Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    plt.tight_layout()
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f"Training history plot saved to: {save_path}")
    plt.close()


def train_model(config_path: str):
    """Loads data, builds model, trains it, and saves the best model and history plot."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    model_cfg = config['model']
    target_col = config['preprocessing']['target_col']

    # Load processed data
    logging.info("Loading processed data...")
    try:
        train_df = data_loader.load_processed_data(paths['train_data'])
        val_df = data_loader.load_processed_data(paths['val_data'])
    except FileNotFoundError:
        logging.error("Processed data files not found. Please run the preprocessing notebook first.")
        return

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}")

    # Calculate class weights for imbalance
    neg, pos = np.bincount(y_train)
    total = neg + pos
    logging.info(f'Training examples:\n    Total: {total}\n    Positive: {pos} ({100 * pos / total:.2f}% of total)\n')

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logging.info(f"Calculated class weights: {class_weight_dict}")

    # Optional: Calculate initial bias for output layer
    initial_bias = np.log([pos / neg])
    logging.info(f"Calculated initial bias: {initial_bias}")


    # Build model
    input_shape = (X_train.shape[1],)
    nn_model = model_builder.build_nn_model(
        input_shape=input_shape,
        dropout_rate=model_cfg['dropout_rate'],
        output_bias=initial_bias # Pass calculated bias
        )

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate'])
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    nn_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logging.info("Model compiled successfully.")

    # Callbacks
    # Ensure directory for saving model exists
    os.makedirs(os.path.dirname(paths['model_save_path']), exist_ok=True)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=paths['model_save_path'], # Keep the .keras extension, TF handles it
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    save_format='tf',  # <-- ADD THIS LINE
    verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',  # Monitor validation AUC
        mode='max',        # Stop when AUC stops improving
        patience=model_cfg['early_stopping_patience'],
        restore_best_weights=True, # Restore weights from the epoch with the best val_auc
        verbose=1
    )

    # Train model
    logging.info("Starting model training...")
    history = nn_model.fit(
        X_train,
        y_train,
        epochs=model_cfg['epochs'],
        batch_size=model_cfg['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, early_stopping],
        class_weight=class_weight_dict, # Use class weights
        verbose=1 # Or 2 for more detailed logs per epoch
    )
    logging.info("Model training finished.")

    # Plot and save training history
    history_plot_path = os.path.join(paths['plot_save_dir'], "training_history.png")
    plot_training_history(history, history_plot_path)

    logging.info(f"Best model saved to: {paths['model_save_path']}") # Checkpoint saves automatically

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the patient readmission prediction model.")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()
    train_model(args.config)
