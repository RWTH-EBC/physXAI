import os
import numpy as np
from physXAI.preprocessing.training_data import TrainingDataMultiStep, copy_and_crop_td_multistep
from physXAI.models.ann.configs.ann_model_configs import RNNModelConstruction_config, MonotonicRNNModelConstruction_config
from physXAI.models.ann.keras_models.keras_models import NonNegPartial, DiagonalPosConstraint, PCNNCell, InputSliceLayer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf


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
    prior_layer = config['prior_layer']
    activation = config['activation']

    # Rescaling for output layer
    rescale_min = float(keras.ops.cast(keras.ops.min(td.y_train), dtype="float32").numpy())
    rescale_max = float(keras.ops.cast(keras.ops.max(td.y_train), dtype="float32").numpy())

    # Input layer
    inputs = keras.Input(shape=(out_steps, num_features))

    # Output rnn model
    if warmup:
        x_train = td.X_train[0]
    else:
        x_train = td.X_train
    o_model = out_model(x_train.reshape(-1, num_features), num_features, rnn_layer, rnn_units, num_outputs,
                        rescale_min, rescale_max, prior_layer=prior_layer, activation=activation)

    # Warmup
    if warmup:
        # Create warmup model
        initial_value_layer = keras.Input(shape=(warmup_width, num_warmup_features))
        int_model = init_model(td.X_train[1].reshape(-1, num_warmup_features), warmup_width, num_warmup_features,
                               init_layer, rnn_layer, rnn_units, 'init_model')
        state = int_model(initial_value_layer)

    # No warmup
    else:
        # Initialize models with zeros
        initial_value_layer = None
        int_model = init_zeros(num_features, rnn_units, out_steps, rnn_layer)
        cropped_inputs = keras.layers.Cropping1D(cropping=(0, out_steps - 1))(inputs)
        state = [int_model(cropped_inputs)]

    # Get output predictions
    prediction, *_ = o_model([inputs, *state])

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
               rnn_layer: str, rnn_units: int, name: str):
    """
    Creates a Keras model to initialize the RNN state using a warmup sequence.

    Args:
        warmup_df (np.ndarray): The warmup sequence data  used to adapt the normalization layer.
                                Shape (samples * warmup_width, num_warmup_features).
        warmup_width (int): Number of time steps in the warmup sequence.
        num_warmup_features (int): Number of features in the warmup sequence.
        init_layer (str): Type of layer to process the warmup sequence ('dense', 'GRU', 'RNN', 'LSTM', 'LastOutput').
        rnn_layer (str): Type of the main RNN layer ('LSTM', 'GRU', 'RNN'), used to determine
                             the number of state tensors needed if init_layer_type is 'dense'.
        rnn_units (int): Number of units for the RNN/Dense layers used in initialization.
        name (str): The name of the returned model.

    Returns:
        keras.Model: A Keras model that takes a warmup sequence and returns the initial RNN state(s).
    """

    # Input layer
    inputs = keras.Input(shape=(warmup_width, num_warmup_features))

    # Normalization layer
    normalization_layer = keras.layers.Normalization()
    normalization_layer.adapt(warmup_df)
    normalized_inputs = normalization_layer(inputs)

    # Init layer
    if init_layer == 'dense':
        dense_init = keras.layers.Dense(units=rnn_units, activation='softplus')
        normalized_inputs = keras.layers.Reshape((warmup_width, num_warmup_features))(normalized_inputs)
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
    elif init_layer == "LastOutput":
        cropped = keras.layers.Cropping1D(cropping=(warmup_width - 1, 0))(normalized_inputs)
        reshaped = keras.layers.Reshape((1, 1))(cropped)
        flattened = keras.layers.Flatten()(reshaped)
        if rnn_layer != "LSTM":
            state = [keras.layers.Dense(rnn_units, trainable=False, use_bias=False, kernel_initializer=keras.initializers.Ones())(flattened)]
        else:
            init_h = keras.layers.Dense(rnn_units, trainable=False, use_bias=False, kernel_initializer=keras.initializers.Ones(), name="init_hidden")(flattened)
            init_c = keras.layers.Dense(rnn_units, trainable=False, use_bias=False, kernel_initializer=keras.initializers.Ones(), name="init_cell")(flattened)
            state = [init_h, init_c]
    else:
        raise NotImplementedError(f'Not implemented {init_layer}')

    return keras.Model(inputs, state, name=name)


def init_zeros(num_features: int, rnn_units: int, out_steps: int, rnn_layer: str):
    """
    Creates a Keras model that generates a zero initial state for an RNN.
    The state's batch size dimension will match the input batch size.

    Args:
        num_features (int): Number of features in the main input sequence (used by the input layer).
        rnn_units (int): The number of units in the RNN, determining the size of the zero state.
        out_steps (int): Number of time steps in the main input sequence.
        rnn_layer (str): Type of the main RNN layer ('LSTM', 'GRU', 'RNN')

    Returns:
        keras.Model: A Keras model that takes a dummy input (main sequence shape) and
                     returns a zero tensor suitable as an initial RNN hidden state.
    """

    initial_value_layer = keras.Input(shape=(1, num_features))
    dense_zeros = keras.layers.Dense(rnn_units, activation='linear', use_bias=False,
                                     kernel_initializer=keras.initializers.Zeros())
    dense_zeros.trainable = False
    zeros = keras.layers.Reshape((1, num_features))(initial_value_layer)
    zeros = keras.layers.Flatten()(zeros)
    zeros = dense_zeros(zeros)
    if rnn_layer == 'LSTM':
        zeros = [zeros, zeros]
    return keras.Model(inputs=initial_value_layer, outputs=zeros, name=f'init_zeros_for_{rnn_layer}')


def out_model(inputs_df: np.ndarray, num_features: int, rnn_layer: str, rnn_units: int, num_outputs: int,
              rescale_min: float, rescale_max: float, prior_layer: str = None, activation: str = 'tanh'):
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
        rescale_min (float): Minimum value used by a Rescaling layer applied to the predictions.
        rescale_max (float): Maximum value used by a Rescaling layer applied to the predictions.
        prior_layer (str): The layer before RNN layer to generate more flexibility of the overall model structure.
        activation (str): The activation function to be used in the out_model, only for RNN.

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
        rnn = keras.layers.SimpleRNN(rnn_units, return_state=True, return_sequences=True, activation=activation)
    elif rnn_layer == "LSTM":
        input_init = [keras.Input(shape=(rnn_units,)) for _ in range(2)]  # List of two inputs for LSTM states
        rnn = keras.layers.LSTM(rnn_units, return_state=True, return_sequences=True)
    else:
        raise NotImplementedError(f'Not implemented {rnn_layer}')

    # Prior layer
    if prior_layer == "dense":
        prior = keras.layers.Dense(rnn_units, activation='softplus')
        rnn_input = prior(normalized_inputs)
    elif prior_layer is None:
        rnn_input = normalized_inputs

    # Predict outputs and states
    pred, *state = rnn(rnn_input, initial_state=input_init)

    # Final dense Layer
    dense = keras.layers.Dense(num_outputs)
    pred = dense(pred)

    # Rescaling layer
    rescaling_layer = keras.layers.Rescaling(scale=rescale_max - rescale_min, offset=rescale_min)
    pred = rescaling_layer(pred)

    return keras.Model([inputs, input_init], [pred, *state], name='out_model')


def MonotonicRNNModelConstruction(config: dict, td: TrainingDataMultiStep):
    """
    Constructs a Monotonic Recurrent Neural Network (MRNN) model using Keras Functional API.
    This type of network can enforce monotonicity constraints on the input features.

    Args:
        config (dict): A dictionary containing the configuration parameters for the MRNN.
                       Validated against `MonotonicRNNModelConstruction_config`.
        td (TrainingDataGeneric): An object containing the training data, used for normalization,
                           input shape, and determining monotonicity constraints based on column names.

    Returns:
        keras.Model: The constructed Keras functional model.
    """

    # Validate the input configuration dictionary and convert it to a dictionary
    config = MonotonicRNNModelConstruction_config.model_validate(config).model_dump()

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
    activation = config['activation']
    monotonicity = config['monotonicity']
    dis_layer = config['dis_layer']
    dis_units = config['dis_units']
    dis_activation = config['dis_activation']
    init_dis = config['init_dis']
    fully_connected = config['fully_connected']

    # Rescaling for output layer
    rescale_min = float(keras.ops.cast(keras.ops.min(td.y_train), dtype="float32").numpy())
    rescale_max = float(keras.ops.cast(keras.ops.max(td.y_train), dtype="float32").numpy())

    # Input layer
    inputs = keras.Input(shape=(out_steps, num_features))

    # Get training data
    if warmup:
        x_train = td.X_train[0]
    else:
        x_train = td.X_train

    # Get monotonicity constraints
    mono = list(monotonicity.values())

    def out_mrnn_model(inputs_df: np.ndarray, num_features: int, rnn_units: int, dis_units: int, num_outputs: int,
                    rescale_min: float, rescale_max: float, rnn_layer: str, dis_layer: str,
                       activation: str,  monotonicity: list, dis_activation: str, fully_connected: bool):
        """
        Creates the main Keras model that processes an input sequence with an initial RNN state
        to produce predictions and the final RNN state.

        Args:
            inputs_df (np.ndarray): The main input sequence data used to adapt the normalization layer.
                                    Shape (samples, steps, features).
            num_features (int): Number of features in the main input sequence.
            rnn_layer (str): Type of monotonic RNN layer to use ('RNN').
            rnn_units (int): Number of units in the monontonic RNN layer.
            dis_units (int): Number of units in the disturbance layer
            num_outputs (int): Number of output features to predict at each time step.
            rescale_min (float): Minimum value used by a Rescaling layer applied to the predictions.
            rescale_max (float): Maximum value used by a Rescaling layer applied to the predictions.
            dis_layer (str): The layer for capturing the influence of disturbance inputs without monotonicity.
            activation (str): The activation function to be used in the monotonic rnn.
            monotonicity (dict[str, int]): Dictionary mapping feature names to monotonicity type
                             (-1 for decreasing, 0 for no constraint, 1 for increasing).
            dis_activation (str): The activation function to be used in the disturbance rnn.
            fully_connected (bool): Whether the hidden states in the main RNN are fully connected with each other.

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

        slice_mono = InputSliceLayer(list(range(0, len(monotonicity))))
        normalized_inputs_mono = slice_mono(normalized_inputs)

        slice_dis = InputSliceLayer(list(range(len(monotonicity), num_features)))
        normalized_inputs_dis = slice_dis(normalized_inputs)

        # Get kernal constraint
        kernal_constraint = NonNegPartial(monotonicity)

        # 1. Monotonic RNN layer
        if rnn_layer == "RNN":
            rnn_input_init = keras.Input(shape=(rnn_units,))
            if fully_connected:
                recurrent_kernal_constraint = keras.constraints.NonNeg()
            else:
                recurrent_kernal_constraint = DiagonalPosConstraint()
            rnn = keras.layers.SimpleRNN(rnn_units, return_state=True, return_sequences=True, activation=activation,
                                         kernel_constraint=kernal_constraint,
                                         recurrent_constraint=recurrent_kernal_constraint)
        else:
            raise NotImplementedError(f'Not implemented {rnn_layer}')

        # Predict outputs and states
        pred_mon, *state_mono = rnn(normalized_inputs_mono, initial_state=rnn_input_init)

        # Final dense Layer
        dense_mono = keras.layers.Dense(num_outputs, kernel_constraint=keras.constraints.NonNeg())
        pred_mono = dense_mono(pred_mon)

        # 2. Disturbance layer
        if dis_layer == "LSTM":
            dis_input_init = [keras.Input(shape=(dis_units,)) for _ in range(2)]
            dis = keras.layers.LSTM(dis_units, return_state=True, return_sequences=True)
            pred_dis, *state_dis = dis(normalized_inputs_dis, initial_state=dis_input_init)
            dense_dis = keras.layers.Dense(num_outputs)
            pred_dis = dense_dis(pred_dis)
        elif dis_layer == 'dense':
            dis = keras.layers.Dense(dis_units, activation=dis_activation)
            pred_dis = dis(normalized_inputs_dis)
            pred_dis = keras.layers.Dense(num_outputs)(pred_dis)
        else:
            raise NotImplementedError(f'Not implemented {dis_layer}')

        # add up to get the prediction
        pred = keras.layers.Add()([pred_mono, pred_dis])

        # Rescaling layer
        rescaling_layer = keras.layers.Rescaling(scale=rescale_max - rescale_min, offset=rescale_min)
        pred = rescaling_layer(pred)

        # construct the keras model
        if dis_layer == 'LSTM':
            mrnn_model = keras.Model([inputs, rnn_input_init, dis_input_init], [pred, *state_mono, *state_dis], name='out_model')
        else:
            mrnn_model = keras.Model([inputs, rnn_input_init], [pred, *state_mono], name='out_model')

        return mrnn_model


    o_model = out_mrnn_model(x_train.reshape(-1, num_features), num_features, rnn_units, dis_units, num_outputs,
                        rescale_min, rescale_max, rnn_layer, dis_layer, activation, mono, dis_activation, fully_connected)

    def init_mrnn_model(warmup: bool, warmup_width, num_warmup_features, out_steps, num_features, init_layer, rnn_layer, rnn_units, init_dis, dis_layer, dis_units, td):

        inputs = keras.layers.Input(shape=(out_steps, num_features))
        initial_value_layer = keras.Input(shape=(warmup_width, num_warmup_features))
        # Warmup
        if warmup:
            # Create warmup model
            int_model_mono = init_model(td.X_train[1].reshape(-1, num_warmup_features), warmup_width, num_warmup_features,
                                   init_layer, rnn_layer, rnn_units, 'init_model_mono')
            state_mono = int_model_mono(initial_value_layer)
            if dis_layer == 'LSTM':
                if init_dis == 'Zero':
                    int_model_dis = init_zeros(num_warmup_features, dis_units, out_steps, dis_layer)
                    cropped_inputs = keras.layers.Cropping1D(cropping=(0, warmup_width - 1))(initial_value_layer)
                    state_dis = int_model_dis(cropped_inputs)
                else:
                    int_model_dis = init_model(td.X_train[1].reshape(-1, num_warmup_features), warmup_width, num_warmup_features,
                                               init_dis, dis_layer, dis_units, 'init_model_dis')
                    state_dis = int_model_dis(initial_value_layer)
            model = keras.Model([initial_value_layer], [state_mono, state_dis], name='init_model')

        # No warmup
        else:
            # Initialize models with zeros
            int_model_mono = init_zeros(num_features, rnn_units, out_steps, rnn_layer)
            cropped_inputs = keras.layers.Cropping1D(cropping=(0, out_steps - 1))(inputs)
            state_mono = [int_model_mono(cropped_inputs)]
            if dis_layer == "LSTM":
                int_model_dis = init_zeros(num_features, dis_units, out_steps, dis_layer)
                cropped_inputs = keras.layers.Cropping1D(cropping=(0, out_steps - 1))(inputs)
                state_dis = int_model_dis(cropped_inputs)
            elif dis_layer == "dense":
                pass
            model = keras.Model([inputs], [state_mono, state_dis], name='init_model')

        return initial_value_layer, model

    initial_value_layer, i_model = init_mrnn_model(warmup, warmup_width, num_warmup_features, out_steps, num_features, init_layer, rnn_layer, rnn_units, init_dis, dis_layer, dis_units, td)
    if warmup:
        state_mono, state_dis = i_model([initial_value_layer])
    else:
        state_mono, state_dis = i_model([inputs])

    # Get output predictions
    if dis_layer == 'LSTM':
        prediction, *_ = o_model([inputs, *state_mono, state_dis])
    elif dis_layer == 'dense':
        prediction, *_ = o_model([inputs, *state_mono])

    # Reshape output
    outputs = keras.layers.Reshape((out_steps, num_outputs))(prediction)

    # Define the model
    if warmup:
        model = keras.Model([inputs, initial_value_layer], outputs)
    else:
        model = keras.Model(inputs, outputs)

    model.summary()
    if warmup:
        i_model.summary()
    o_model.summary()

    return model

def PCNNModelConstruction(config: dict, disturbance_ann, td: TrainingDataMultiStep, non_lin_ann=None):
    """
    Constructs a Physically Consistent Neural Network (PCNN) for multi-step time series forecasting.

    Args:
        config (dict): A dictionary containing configuration parameters for the RNN model.
                       Validated against `RNNModelConstruction_config`.
        disturbance_ann (ANNModel): The disturbance ANN for the disturbance module of the PCNN.
        td (TrainingDataMultiStep): An object containing the multi-step training data.
                                    It provides shapes for inputs, outputs, and warmup sequences.
        non_lin_ann (ANNModel): An additional ANN to fully capture non-linear input dynamics of inputs that are then fed into the linear module
    Returns:
        keras.Model: The constructed Keras functional model for RNN-based forecasting.
    """

    dis_inputs = config['dis_inputs']
    # solely feed disturbance ann with disturbance training data too initialize correct dimension
    cropped_dis_td = copy_and_crop_td_multistep(td, dis_inputs)
    disturbance_ann_keras = disturbance_ann.generate_model(td=cropped_dis_td)

    non_lin_inputs = config['non_lin_inputs']
    if non_lin_inputs is not None:
        assert non_lin_ann is not None, ("If non-linear inputs are given for the linear modul, an additional ANN has to"
                                         "be given as well to capture the non-linear dynamics appropriately")
        cropped_non_lin_td = copy_and_crop_td_multistep(td, -non_lin_inputs)
        non_lin_ann_keras = non_lin_ann.generate_model(td=cropped_non_lin_td)
    else:
        non_lin_ann_keras = None

    assert isinstance(td.X_train, tuple), "PCNN needs a warmup-width of 1 to initialize the disturbance state"
    warmup = True
    out_steps = td.X_train[0].shape[1]
    warmup_width = td.X_train[1].shape[1]
    num_features = td.X_train[0].shape[2]
    num_warmup_features = td.X_train[1].shape[2]

    num_outputs = td.y_train.shape[2]

    # Get config
    rnn_units = config['rnn_units']
    rnn_layer = config['rnn_layer']

    # Rescaling for output layer
    rescale_min = keras.ops.cast(keras.ops.min(td.y_train), dtype="float32")
    rescale_max = keras.ops.cast(keras.ops.max(td.y_train), dtype="float32")

    # Input layer
    inputs = keras.Input(shape=(out_steps, num_features))

    def out_pcnn_model(inputs_df: np.ndarray, num_features: int, rnn_units: int, num_outputs: int,
              rescale_min: float, rescale_max: float):

        # Input layer
        inputs = keras.Input(shape=(None, num_features), name='RNN_input')

        # Normalization layer
        normalization_layer = keras.layers.Normalization(name='RNN_normalization')
        normalization_layer.adapt(inputs_df)
        normalized_inputs = normalization_layer(inputs)

        #input_init = [keras.Input(shape=(rnn_units,)) for _ in range(2)]
        input_init = keras.Input(shape=(2,), name='RNN_hidden_input')

        # Define the RNN layer using PCNNCell properly.
        rnn_cell_instance = PCNNCell(dis_ann=disturbance_ann_keras, dis_inputs=dis_inputs,
                                     non_lin_ann=non_lin_ann_keras, non_lin_inputs=non_lin_inputs)
        rnn_layer = keras.layers.RNN(rnn_cell_instance, return_state=True, return_sequences=True, name="pcnn")

        # Call the RNN layer with normalized inputs and initial state.
        predictions_and_states = rnn_layer(normalized_inputs, initial_state=input_init)

        # Unpack predictions and states correctly.
        pred, *state = predictions_and_states

        return keras.Model([inputs, input_init], [pred, *state], name='out_model')

    # Output pcnn model
    o_model = out_pcnn_model(td.X_train[0].reshape(-1, num_features), num_features, rnn_units, num_outputs,
                        rescale_min, rescale_max)

    def init_pcnn(num_warmup_features: int) -> keras.Model:
        """ Generate initial model state: D_0 = TAirRoom, E_0 = 0 """

        initial_value_layer = keras.Input(shape=(num_warmup_features,))
        dense_ones = keras.layers.Dense(1, activation='linear', use_bias=False,
                                        kernel_initializer=keras.initializers.Ones(), trainable=False)(initial_value_layer)
        dense_zeros = keras.layers.Dense(1, activation='linear', use_bias=False,
                                         kernel_initializer=keras.initializers.Zeros(), trainable=False)(initial_value_layer)
        concat = keras.layers.Concatenate(axis=-1)([dense_ones, dense_zeros])

        return keras.Model(inputs=initial_value_layer, outputs=concat, name='init_model')

    # Create warmup model
    initial_value_layer = keras.Input(shape=(num_warmup_features,))
    int_model = init_pcnn(num_warmup_features)
    state = int_model(initial_value_layer)

    #int_model = init_zeros(num_features, 2, out_steps)
    #state = [int_model(inputs)]

    # Get output predictions
    prediction, *_ = o_model([inputs, state])

    # Reshape output
    outputs = keras.layers.Reshape((out_steps, num_outputs))(prediction)

    # Define the model
    model = keras.Model([inputs, initial_value_layer], outputs)

    model.summary(show_trainable=True)
    int_model.summary(show_trainable=True)
    o_model.summary(show_trainable=True)

    return model
