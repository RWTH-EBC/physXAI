import os
from physXAI.preprocessing.training_data import TrainingDataGeneric
from physXAI.models.ann.configs.ann_model_configs import (ClassicalANNConstruction_config,
                                                                 CMNNModelConstruction_config)
from physXAI.models.ann.keras_models.keras_models import NonNegPartial, ConcaveActivation, SaturatedActivation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def ClassicalANNConstruction(config: dict, td: TrainingDataGeneric):
    """
    Constructs a classical Artificial Neural Network (ANN) model using Keras.

    Args:
        config (dict): A dictionary containing the configuration parameters for the ANN.
                       This will be validated against `ClassicalANNConstruction_config`.
        td (TrainingDataGeneric): An object containing the training data,
                           used for adapting normalization and determining input/output shapes.

    Returns:
        keras.Model: The constructed Keras sequential model.
    """

    # Validate the input configuration dictionary using the Pydantic model and convert it to a dictionary
    config = ClassicalANNConstruction_config.model_validate(config).model_dump()

    # Get config
    n_layers = config['n_layers']
    n_neurons = config['n_neurons']
    # If n_neurons is a single integer, replicate it for all layers
    if isinstance(n_neurons, int):
        n_neurons = [n_neurons] * n_layers
    else:
        assert len(n_neurons) == n_layers
    n_featues = td.X_train_single.shape[1]
    activation_function = config['activation_function']
    # If activation_function is a single string, replicate it for all layers
    if isinstance(activation_function, str):
        activation_function = [activation_function] * n_layers
    else:
        assert len(activation_function) == n_layers

    # Rescaling for output layer
    rescale_min = float(td.y_train_single.min())
    rescale_max = float(td.y_train_single.max())

    # Build artificial neural network as Sequential
    model = keras.Sequential()

    # Add input layer
    model.add(keras.layers.Input(shape=(n_featues,)))

    # Add normalization layer
    normalization = keras.layers.Normalization()
    normalization.adapt(td.X_train_single)
    model.add(normalization)

    for i in range(0, n_layers):
        # For each layer add dense
        model.add(keras.layers.Dense(n_neurons[i], activation=activation_function[i]))
    # Add output layer
    model.add(keras.layers.Dense(1, activation='linear'))
    # Add rescaling
    if config['rescale_output']:
        model.add(keras.layers.Rescaling(scale=rescale_max - rescale_min, offset=rescale_min))

    model.summary()

    return model


def CMNNModelConstruction(config: dict, td: TrainingDataGeneric):
    """
    Constructs a Constrained Monotonic Neural Network (CMNN) model using Keras Functional API.
    This type of network can enforce monotonicity constraints on the input features.

    Args:
        config (dict): A dictionary containing the configuration parameters for the CMNN.
                       Validated against `CMNNModelConstruction_config`.
        td (TrainingDataGeneric): An object containing the training data, used for normalization,
                           input shape, and determining monotonicity constraints based on column names.

    Returns:
        keras.Model: The constructed Keras functional model.
    """

    # Validate the input configuration dictionary and convert it to a dictionary
    config = CMNNModelConstruction_config.model_validate(config).model_dump()

    # Get config
    n_layers = config['n_layers']
    n_neurons = config['n_neurons']
    # If n_neurons is a single integer, replicate it for all layers
    if isinstance(n_neurons, int):
        n_neurons = [n_neurons] * n_layers
    else:
        assert len(n_neurons) == n_layers
    n_featues = td.X_train_single.shape[1]
    activation_function = config['activation_function']
    # If activation_function is a single string, replicate it for all layers
    if isinstance(activation_function, str):
        activation_function = [activation_function] * n_layers
    else:
        assert len(activation_function) == n_layers

    # Get monotonicity constraints
    mono = config['monotonicities']
    if mono is None:
        monotonicities = [0] * n_featues
    else:
        monotonicities = [0 if name not in mono.keys() else mono[name] for name in td.columns]

    # Rescaling for output layer
    rescale_min = float(td.y_train_single.min())
    rescale_max = float(td.y_train_single.max())

    # Add input layer
    input_layer = keras.layers.Input(shape=(n_featues,))

    # Add normalization layer
    normalization = keras.layers.Normalization()
    normalization.adapt(td.X_train_single)
    x = normalization(input_layer)

    # Add dense layer
    activation_split = config['activation_split']
    # Determine activation split
    if activation_split is None:
        if mono is None:
            activation_split = [1, 0, 0]
        else:
            activation_split = [1, 1, 1]
    # First layer has partial constraints based on monotonicities
    kernel_contraint = NonNegPartial(monotonicities)
    for i in range(0, n_layers):
        x_split = list()
        # Convex activation
        if activation_split[0] > 0:
            x1 = keras.layers.Dense(int(n_neurons[i] * activation_split[0] / sum(activation_split)),
                                    activation=activation_function[i], kernel_constraint=kernel_contraint)(x)
            x_split.append(x1)
        # Concave activation
        if activation_split[1] > 0:
            x2 = keras.layers.Dense(int(n_neurons[i] * activation_split[1] / sum(activation_split)),
                                    activation=ConcaveActivation(activation_function[i]),
                                    kernel_constraint=kernel_contraint)(x)
            x_split.append(x2)
        # Saturated activation
        if activation_split[2] > 0:
            x3 = keras.layers.Dense(int(n_neurons[i] * activation_split[2] / sum(activation_split)),
                                    activation=SaturatedActivation(activation_function[i]),
                                    kernel_constraint=kernel_contraint)(x)
            x_split.append(x3)
        # Concatenate activations
        if len(x_split) > 1:
            x = keras.layers.concatenate(x_split)
        else:
            x = x_split[0]

        # after monotonicity constraint was applied,
        # in all following layers the weights have to be non-neg to maintain the monotonicity
        kernel_contraint = keras.constraints.NonNeg()

    # Add output layer
    x = keras.layers.Dense(1, activation='linear', kernel_constraint=keras.constraints.NonNeg())(x)

    # Add rescaling
    if config['rescale_output']:
        x = keras.layers.Rescaling(scale=rescale_max - rescale_min, offset=rescale_min)(x)

    # # Add min / max constraints
    # min_value = config['min_value']
    # max_value = config['max_value']
    # if min_value is not None or max_value is not None:
    #     d = keras.layers.Dense(1, activation=LimitedActivation(max_value, min_value),
    #                            kernel_initializer=keras.initializers.Ones(), use_bias=False)
    #     d.trainable = False
    #     x = d(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)

    model.summary()

    return model
