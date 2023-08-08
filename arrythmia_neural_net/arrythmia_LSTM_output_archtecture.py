import tensorflow as tf
import keras
from keras.layers import Input, LSTM, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam

def create_output_model(input_shape, lstm_units, dropout_rate, optimizer, loss, metrics):
    """ Create a multi-layer LSTM classification model.

    Args:
        input_shape (tuple): Shape of the input data.
        lstm_units (int): Number of units for the LSTM layers.
        dropout_rate (float): Proportion of neurons to drop in the Dropout layers.
        optimizer (str or keras.optimizers.Optimizer): Optimizer for training.
        loss (str or keras.losses.Loss): Loss function for training.
        metrics (list[str or keras.metrics.Metric]): Metrics for training.

    Returns:
        keras.models.Model: The compiled LSTM model.
    """

    # Define inputs
    inputs = Input(shape=input_shape)

    # First LSTM layer
    x = LSTM(units=lstm_units[0], return_sequences=True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev=1), kernel_regularizer=tf.keras.regularizers.l2())(inputs)
    x = Dropout(dropout_rate)(x)

    # Second LSTM layer
    x = LSTM(units=lstm_units[1], return_sequences=True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev=1), kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = Dropout(dropout_rate)(x)

    
    output = tf.keras.layers.GlobalAveragePooling1D()(x)
    #output = tf.keras.layers.Flatten()(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
