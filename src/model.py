import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Tuple
import numpy as np # Make sure numpy is imported if not already

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_nn_model(input_shape: Tuple[int], dropout_rate: float = 0.3, output_bias=None) -> keras.Model: # Argument 'output_bias' is the initial numpy value
    """
    Builds a simple Feedforward Neural Network model using Keras Sequential API.

    Args:
        input_shape (Tuple[int]): The shape of the input features (number of features,).
        dropout_rate (float): Dropout rate for regularization.
        output_bias (Optional[np.ndarray]): Initial bias value (as numpy array) for the output layer.

    Returns:
        keras.Model: The compiled Keras model.
    """
    logging.info(f"Building NN model with input shape: {input_shape}")

    keras_bias_initializer = None # Default initializer if no bias is provided
    if output_bias is not None:
        #    Access the scalar value if it's a single-element array
        bias_value_to_log = output_bias[0] if isinstance(output_bias, np.ndarray) and output_bias.size == 1 else output_bias
        try:
            logging.info(f"Using initial output bias value: {bias_value_to_log:.4f}")
        except TypeError: # Handle cases where formatting might fail
             logging.info(f"Using initial output bias value: {bias_value_to_log}")

        keras_bias_initializer = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="input_layer"),
            layers.Dense(128, activation="relu", name="dense_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            layers.Dense(64, activation="relu", name="dense_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),
            # Use the Keras initializer object here (which might be None or the Constant object)
            layers.Dense(1, activation="sigmoid", bias_initializer=keras_bias_initializer, name="output_layer"),
        ]
    )

    logging.info("NN model built successfully.")
    model.summary(print_fn=logging.info) # Log model summary

    return model
