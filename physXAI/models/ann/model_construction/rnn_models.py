import os
import numpy as np
from physXAI.preprocessing.training_data import TrainingDataMultiStep
from physXAI.models.ann.configs.ann_model_configs import RNNModelConstruction_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def RNNModelConstruction(config: dict, td: TrainingDataMultiStep):
    """
    Constructs a Recurrent Neural Network (RNN) model for multi-step time series forecasting.
    The model can optionally use a "warmup" sequence to initialize the RNN's hidden state.

    Args:
        config (dict): A dictionary containing configuration parameters for the RNN model.
                       Validated against `RNNModelConstruction_config`.
        td (TrainingDataMultiStep): An object containing the multi-step training data.
                                    It provides shapes for inputs, outputs, and warmup sequences.

    Returns:
        keras.Model: The constructed Keras functional model for RNN-based forecasting.
    """

    # Validate the input configuration dictionary using the Pydantic model and convert it to a dictionary
    config = RNNModelConstruction_config.model_validate(config).model_dump()

    # Get boundary conditions from training data
    # With initialization data
    if isinstance(td.X_train, tuple):
        warmup = True
        out_steps = td.X_train[0].shape[1]
        warmup_width = td.X_train[1].shape[1]
        num_features = td.X_train[0].shape[2]
        num_warmup_features = td.X_train[1].shape[2]
    # Without initialization data
    else:
        warmup = False
        out_steps = td.X_train.shape[1]
        warmup_width = 0
        num_features = td.X_train.shape[2]
        num_warmup_features = 0
    num_outputs = td.y_train.shape[2]

    # Get config
    rnn_units = config['rnn_units']
    init_layer = config['init_layer']
    rnn_layer = config['rnn_layer']

    # Rescaling for output layer
    rescale_mean = keras.ops.cast(keras.ops.mean(td.y_train), dtype="float32")
    rescale_sigma = keras.ops.cast(keras.ops.std(td.y_train), dtype="float32")

    # Input layer
    inputs = keras.Input(shape=(out_steps, num_features))

    # Output rnn model
    o_model = out_model(td.X_train[0].reshape(-1, num_features), num_features, rnn_layer, rnn_units,  num_outputs,
                        rescale_mean, rescale_sigma)

    # Warmup
    if warmup:
        # Create warmup model
        initial_value_layer = keras.Input(shape=(warmup_width, num_warmup_features))
        int_model = init_model(td.X_train[1].reshape(-1, num_warmup_features), warmup_width, num_warmup_features,
                               init_layer, rnn_layer, rnn_units)
        state = int_model(initial_value_layer)

    # No warmup
    else:
        # Initialize models with zeros
        initial_value_layer = None
        int_model = init_zeros(num_features, rnn_units, out_steps)
        if rnn_layer == "LSTM":
            state = [int_model(inputs), int_model(inputs)]
        else:
            state = [int_model(inputs)]

    # Get output predictions
    prediction, *_ = o_model([inputs, state])

    # Reshape output
    outputs = keras.layers.Reshape((out_steps, num_outputs))(prediction)

    # Define the model
    if warmup:
        model = keras.Model([inputs, initial_value_layer], outputs)
    else:
        model = keras.Model(inputs, outputs)

    model.summary()
    if warmup:
        int_model.summary()
    o_model.summary()

    return model


def init_model(warmup_df: np.ndarray, warmup_width: int, num_warmup_features: int, init_layer: str,
               rnn_layer: str, rnn_units: int):
    """
    Creates a Keras model to initialize the RNN state using a warmup sequence.

    Args:
        warmup_df (np.ndarray): The warmup sequence data  used to adapt the normalization layer.
                                Shape (samples, warmup_width, num_warmup_features).
        warmup_width (int): Number of time steps in the warmup sequence.
        num_warmup_features (int): Number of features in the warmup sequence.
        init_layer (str): Type of layer to process the warmup sequence ('dense', 'GRU', 'RNN', 'LSTM').
        rnn_layer (str): Type of the main RNN layer ('LSTM', 'GRU', 'RNN'), used to determine
                             the number of state tensors needed if init_layer_type is 'dense'.
        rnn_units (int): Number of units for the RNN/Dense layers used in initialization.

    Returns:
        keras.Model: A Keras model that takes a warmup sequence and returns the initial RNN state(s).
    """

    # Input layer
    inputs = keras.Input(shape=(warmup_width, num_warmup_features))

    # Normalization layer
    normalization_layer = keras.layers.Normalization()
    normalization_layer.adapt(warmup_df)
    normalized_inputs = normalization_layer(inputs)
    normalized_inputs = keras.layers.Reshape((warmup_width, num_warmup_features))(normalized_inputs)

    # Init layer
    if init_layer == 'dense':
        dense_init = keras.layers.Dense(units=rnn_units, activation='softplus')
        normalized_inputs = keras.layers.Flatten()(normalized_inputs)
        if rnn_layer == 'LSTM':  # For LSTM, creating two Dense layers
            dense_init2 = keras.layers.Dense(units=rnn_units, activation='softplus')
            state = [dense_init(normalized_inputs), dense_init2(normalized_inputs)]
        else:
            state = [dense_init(normalized_inputs)]
    elif init_layer == 'GRU':
        rnn_init = keras.layers.GRU(units=rnn_units, return_state=True)
        _, *state = rnn_init(normalized_inputs)
    elif init_layer == 'RNN':
        rnn_init = keras.layers.SimpleRNN(units=rnn_units, return_state=True)
        _, *state = rnn_init(normalized_inputs)
    elif init_layer == "LSTM":
        rnn_init = keras.layers.LSTM(units=rnn_units, return_state=True)
        _, *state = rnn_init(normalized_inputs)
    else:
        raise NotImplementedError(f'Not implemented {init_layer}')

    return keras.Model(inputs, state, name='init_model')


def init_zeros(num_features: int, rnn_units: int, out_steps: int):
    """
    Creates a Keras model that generates a zero initial state for an RNN.
    The state's batch size dimension will match the input batch size.

    Args:
        num_features (int): Number of features in the main input sequence (used by the input layer).
        rnn_units (int): The number of units in the RNN, determining the size of the zero state.
        out_steps (int): Number of time steps in the main input sequence.

    Returns:
        keras.Model: A Keras model that takes a dummy input (main sequence shape) and
                     returns a zero tensor suitable as an initial RNN hidden state.
    """
    initial_value_layer = keras.Input(shape=(out_steps, num_features))
    crop = keras.layers.Cropping1D(cropping=(0, out_steps-1))
    dense_zeros = keras.layers.Dense(rnn_units, activation='linear', use_bias=False,
                                     kernel_initializer=keras.initializers.Zeros())
    dense_zeros.trainable = False
    cropped = crop(initial_value_layer)
    zeros = keras.layers.Reshape((1, num_features))(cropped)
    zeros = keras.layers.Flatten()(zeros)
    zeros = dense_zeros(zeros)
    return keras.Model(inputs=initial_value_layer, outputs=zeros, name='init_zeros')


def out_model(inputs_df: np.ndarray, num_features: int, rnn_layer: str, rnn_units: int, num_outputs: int,
              rescale_mean: float, rescale_sigma: float):
    """
    Creates the main Keras model that processes an input sequence with an initial RNN state
    to produce predictions and the final RNN state.

    Args:
        inputs_df (np.ndarray): The main input sequence data used to adapt the normalization layer.
                                Shape (samples, steps, features).
        num_features (int): Number of features in the main input sequence.
        rnn_layer (str): Type of RNN layer to use ('GRU', 'RNN', 'LSTM').
        rnn_units (int): Number of units in the RNN layer.
        num_outputs (int): Number of output features to predict at each time step.
        rescale_mean (float): Mean value used by a Rescaling layer applied to the predictions.
        rescale_sigma (float): Standard deviation used by a Rescaling layer applied to the predictions.

    Returns:
        keras.Model: A Keras model that takes [main_input_sequence, initial_state(s)]
                     and returns [prediction_sequence, final_state(s)].
    """
    # Input layer
    inputs = keras.Input(shape=(None, num_features))

    # Normalization layer
    normalization_layer = keras.layers.Normalization()
    normalization_layer.adapt(inputs_df)
    normalized_inputs = normalization_layer(inputs)

    # RNN layer
    if rnn_layer == "GRU":
        input_init = keras.Input(shape=(rnn_units,))
        rnn = keras.layers.GRU(rnn_units, return_state=True, return_sequences=True)
    elif rnn_layer == "RNN":
        input_init = keras.Input(shape=(rnn_units,))
        rnn = keras.layers.SimpleRNN(rnn_units, return_state=True, return_sequences=True)
    elif rnn_layer == "LSTM":
        input_init = [keras.Input(shape=(rnn_units,)) for _ in range(2)]  # List of two inputs for LSTM states
        rnn = keras.layers.LSTM(rnn_units, return_state=True, return_sequences=True)
    else:
        raise NotImplementedError(f'Not implemented {rnn_layer}')

    # Predict outputs and states
    pred, *state = rnn(normalized_inputs, initial_state=input_init)

    # Final dense Layer
    dense = keras.layers.Dense(num_outputs)
    pred = dense(pred)

    # Rescaling layer
    rescaling_layer = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)
    pred = rescaling_layer(pred)

    return keras.Model([inputs, input_init], [pred, *state], name='out_model')
