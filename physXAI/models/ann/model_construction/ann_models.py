import os
import numpy as np
import tensorflow as tf
from physXAI.preprocessing.training_data import TrainingDataGeneric
from physXAI.models.ann.configs.ann_model_configs import (ClassicalANNConstruction_config,
                                                                 CMNNModelConstruction_config,
                                                                 RC1R1CConstruction_config,
                                                                 RC2R2CPhysNetConstruction_config,
                                                                 RC2R2CGokhalePhysNetConstruction_config,
                                                                 RC2R2CGokhalePhysNetWallDynamicsConstruction_config)
from physXAI.models.ann.keras_models.keras_models import NonNegPartial, ConcaveActivation, SaturatedActivation, InputSliceLayer
from physXAI.models.ann.pinn_new.rc_layers import RC1R1CLayer, RC2R2CPhysNetLayer, RC2R2CGokhalePhysNetLayer, RC2R2CGokhalePhysNetWallDynamicsLayer
from physXAI.models.ann.pinn_new.feature_index import _resolve_feature_indices
from physXAI.models.ann.pinn_new.calculate_wall_temperature import calculate_wall_temperature_for_scaling
from physXAI.models.ann.pinn_new.pinn_keras_models import RC1R1CKerasModel, RC2R2CPhysNetKerasModel, RC2R2CGokhalePhysNetKerasModel, RC2R2CGokhalePhysNetWallDynamicsKerasModel
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
    if config['n_features'] is not None:
        n_features = config['n_features']
    else:
        n_features = td.X_train_single.shape[1]
    activation_function = config['activation_function']
    # If activation_function is a single string, replicate it for all layers
    if isinstance(activation_function, str):
        activation_function = [activation_function] * n_layers
    else:
        assert len(activation_function) == n_layers

    # Build artificial neural network as Sequential
    model = keras.Sequential()

    # Add input layer
    model.add(keras.layers.Input(shape=(n_features,)))

    # Add normalization layer
    if config['normalize']:
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
        # Rescaling for output layer
        rescale_mean = float(np.mean(td.y_train_single))
        rescale_sigma = float(np.std(td.y_train_single, ddof=1))
        model.add(keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean))

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
    if config['n_features'] is not None:
        n_features = config['n_features']
    else:
        n_features = td.X_train_single.shape[1]
    activation_function = config['activation_function']
    # If activation_function is a single string, replicate it for all layers
    if isinstance(activation_function, str):
        activation_function = [activation_function] * n_layers
    else:
        assert len(activation_function) == n_layers

    # Get monotonicity constraints
    mono = config['monotonicities']
    if mono is None:
        monotonicities = [0] * n_features
    else:
        monotonicities = [0 if name not in mono.keys() else mono[name] for name in td.columns]

    # Add input layer
    input_layer = keras.layers.Input(shape=(n_features,))

    # Add normalization layer
    if config['normalize']:
        normalization = keras.layers.Normalization()
        normalization.adapt(td.X_train_single)
        x = normalization(input_layer)
    else:
        x = input_layer

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
        # Rescaling for output layer
        rescale_mean = float(np.mean(td.y_train_single))
        rescale_sigma = float(np.std(td.y_train_single, ddof=1))
        x = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)(x)

    # # Add min / max constraints
    # min_value = config['min_value']
    # max_value = config['max_value']
    # if min_value is not None or max_value is not None:
    #     d = keras.layers.Dense(1, activation=LimitedActivation(max_value, min_value),
    #                            kernel_initializer=keras.initializers.Ones(), use_bias=False)
    #     d.trainable = False
    #     x = d(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)

    return model


def RC1R1CConstruction(config: dict, td: TrainingDataGeneric):
    """
    
    """
    # Validate the input configuration dictionary and convert it to a dictionary
    config = RC1R1CConstruction_config.model_validate(config).model_dump()

    # Get config
    n_layers = config['n_layers']
    n_neurons = config['n_neurons']

    if isinstance(n_neurons, int):
        n_neurons = [n_neurons] * n_layers
    else:
        assert len(n_neurons) == n_layers
    if config['n_features'] is not None:
        n_features = config['n_features']
    else:
        n_features = td.X_train_single.shape[1]
    activation_function = config['activation_function']

    if isinstance(activation_function, str):
        activation_function = [activation_function] * n_layers
    else:
        assert len(activation_function) == n_layers

    # Get feature indices for the 1R1C model
    if isinstance(config['t_air_column'], str):
        t_air_index = list(td.columns).index(config['t_air_column'])
    else:
        t_air_index = config['t_air_column']

    # Inject t_toom_index into rc_kwargs if not present
    rc_kwargs = config['rc_kwargs']
    if 't_air_index' not in rc_kwargs:
        rc_kwargs['t_air_index'] = t_air_index


    # -------------------------------------------------------------------------
    # neural core model
    # -------------------------------------------------------------------------
    core_input = keras.layers.Input(shape=(n_features,), name='core_input')

    # Add normalization layer
    if config['normalize']:
        normalization = keras.layers.Normalization()
        normalization.adapt(td.X_train_single)
        x = normalization(core_input)
    else:
        x = core_input

    for i in range(0, n_layers):
        # For each layer add dense
        x = keras.layers.Dense(n_neurons[i], activation=activation_function[i])(x)

    # Add output layer
    output_name = 'change_t_air_dense' if config['predict_delta'] else 't_air_dense'
    y_core = keras.layers.Dense(1, activation='linear', name=output_name)(x)

    # Add rescaling
    if config['rescale_output']:
        # Rescaling for output layer
        rescale_mean = float(np.mean(td.y_train_single))
        rescale_sigma = float(np.std(td.y_train_single, ddof=1))
        y_core = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)(y_core)

    core_model = keras.models.Model(inputs=core_input, outputs=y_core, name='core_model')


    # -------------------------------------------------------------------------
    # PINN
    # -------------------------------------------------------------------------
    pinn_input = keras.layers.Input(shape=(n_features,), name='pinn_input')
    
    y_nn = core_model(pinn_input)

    # -------------------------------------------------------------------------
    # 1R1C physics branch
    # -------------------------------------------------------------------------
    physics_layer = RC1R1CLayer(
        trainable_rc=config["trainable_rc"],
        use_internal_gains=config["use_internal_gains"],
        **rc_kwargs,
    )

    model = RC1R1CKerasModel(
        inputs=pinn_input,
        outputs=y_nn,
        core_model=core_model,
        physics_layer=physics_layer,
        physics_loss_weight=config["physics_loss_weight"],
        predict_delta=config["predict_delta"],
        t_air_index=t_air_index,
        name="pinn_1r1c",
    )

    return model


def RC2R2CPhysNetConstruction(config: dict, td: TrainingDataGeneric):
    """
    
    """
    config = RC2R2CPhysNetConstruction_config.model_validate(config).model_dump()

    encoder_layers = config['encoder_layers']
    encoder_neurons = config['encoder_neurons']
    if isinstance(encoder_neurons, int):
        encoder_neurons= [encoder_neurons] * encoder_layers
    else:
        assert len(encoder_neurons) == encoder_layers

    dynamic_layers = config['dynamic_layers']
    dynamic_neurons = config['dynamic_neurons']
    if isinstance(dynamic_neurons, int):
        dynamic_neurons= [dynamic_neurons] * dynamic_layers
    else:
        assert len(dynamic_neurons) == dynamic_layers

    if config['n_features'] is not None:
        n_features = config['n_features']
    else:
        n_features = td.X_train_single.shape[1]

    activation_function = config['activation_function']
    if isinstance(activation_function, str):
        activation_function = [activation_function] * (encoder_layers + dynamic_layers)
    else:
        assert len(activation_function)== (encoder_layers + dynamic_layers)

    encoder_activation = activation_function[:encoder_layers]
    dynamic_activation = activation_function[encoder_layers:]

    if isinstance(config['t_air_column'], str):
        t_air_index = list(td.columns).index(config['t_air_column'])
    else:
        t_air_index = config['t_air_column']
    

    rc_kwargs = dict(config['rc_kwargs'])
    rc_kwargs['t_air_index'] = t_air_index
    rc_kwargs['predict_delta'] = config['predict_delta']

    prediction_loss_scale = float(np.std(td.y_train_single, ddof=1))
    #prediction_loss_scale=1.0

    t_wall_train = calculate_wall_temperature_for_scaling(
        x=td.X_train_single,
        measured_change_t_air=td.y_train_single,
        columns=td.columns,
        rc_kwargs=rc_kwargs
    )

    physics_loss_scale = float(np.std(t_wall_train, ddof=1))
    #physics_loss_scale=1.0

    encoder_indices = _resolve_feature_indices(config['encoder_features'], td.columns)
    dynamic_indices = _resolve_feature_indices(config['dynamic_features'], td.columns)


    # -------------------------------------------------------------------------
    # neural core model
    # -------------------------------------------------------------------------
    core_input = keras.layers.Input(shape=(n_features,), name='core_input')

    x_encoder_input = InputSliceLayer(feature_indices=encoder_indices, name='encoder_input_slice')(core_input)
    x_dynamic_input = InputSliceLayer(feature_indices=dynamic_indices, name='dynamic_input_slice')(core_input)

    # Add normalization layer
    if config['normalize']:
        encoder_normalization = keras.layers.Normalization()
        encoder_normalization.adapt(td.X_train_single[:,encoder_indices])
        x_encoder = encoder_normalization(x_encoder_input)

        dynamic_normalization = keras.layers.Normalization()
        dynamic_normalization.adapt(td.X_train_single[:,dynamic_indices])
        x_dynamic = dynamic_normalization(x_dynamic_input)
    else:
        x_encoder = x_encoder_input
        x_dynamic = x_dynamic_input

    # -------------------------------------------------------------------------
    # Encoder branch
    # -------------------------------------------------------------------------
    for i in range(0, encoder_layers):
        x_encoder = keras.layers.Dense(encoder_neurons[i], activation=encoder_activation[i])(x_encoder)

    z_latent_core = keras.layers.Dense(1, activation='linear', name='t_wall_latent_dense')(x_encoder)

    # -------------------------------------------------------------------------
    # Dynamic branch
    # -------------------------------------------------------------------------
    x_dynamic = keras.layers.Concatenate()([x_dynamic, z_latent_core])
    for i in range(0, dynamic_layers):
        # For each layer add dense
        x_dynamic = keras.layers.Dense(dynamic_neurons[i], activation=dynamic_activation[i])(x_dynamic)

    # Add output layer
    output_name = 'change_t_air_dense' if config['predict_delta'] else 't_air_dense'
    y_core = keras.layers.Dense(1, activation='linear', name=output_name)(x_dynamic)

    if config['normalize']:
        z_rescale_mean = float(np.mean(td.X_train_single[:, [t_air_index]]))
        z_rescale_sigma = float(np.std(td.X_train_single[:, [t_air_index]], ddof=1))

        #z_rescale_mean = float(np.mean(t_wall_train))
        #z_rescale_sigma = float(np.std(t_wall_train, ddof=1))

        z_latent_core = keras.layers.Rescaling(scale=z_rescale_sigma, offset=z_rescale_mean)(z_latent_core)

    # Add rescaling
    if config['rescale_output']:
        # Rescaling for output layer
        rescale_mean = float(np.mean(td.y_train_single))
        rescale_sigma = float(np.std(td.y_train_single, ddof=1))
        y_core = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)(y_core)

    core_model = keras.models.Model(inputs=core_input, outputs=[y_core, z_latent_core], name='core_model')

    # -------------------------------------------------------------------------
    # PINN
    # -------------------------------------------------------------------------
    pinn_input = keras.layers.Input(shape=(n_features,), name='pinn_input')

    y_nn, _ = core_model(pinn_input)

    # -------------------------------------------------------------------------
    # 2R2C physics branch
    # -------------------------------------------------------------------------
    physics_layer = RC2R2CPhysNetLayer(
        trainable_rc=config["trainable_rc"],
        use_internal_gains=config["use_internal_gains"],
        **rc_kwargs,
    )

    model = RC2R2CPhysNetKerasModel(
        inputs=pinn_input,
        outputs=y_nn,
        core_model=core_model,
        physics_layer=physics_layer,
        physics_loss_weight=config["physics_loss_weight"],
        prediction_loss_scale=prediction_loss_scale,
        physics_loss_scale=physics_loss_scale,
        name="pinn_2r2c",
    )
    
    return model



def RC2R2CGokhalePhysNetConstruction(config: dict, td: TrainingDataGeneric):
    """
    
    """
    config = RC2R2CGokhalePhysNetConstruction_config.model_validate(config).model_dump()

    if isinstance(config['t_air_column'], str):
        t_air_index = list(td.columns).index(config['t_air_column'])
    else:
        t_air_index = config['t_air_column']

    rc_kwargs = dict(config['rc_kwargs'])
    rc_kwargs['t_air_index'] = t_air_index
    rc_kwargs['predict_delta'] = config['predict_delta']

    encoder_layers = config['encoder_layers']
    encoder_neurons = config['encoder_neurons']
    if isinstance(encoder_neurons, int):
        encoder_neurons= [encoder_neurons] * encoder_layers
    else:
        assert len(encoder_neurons) == encoder_layers

    dynamic_layers = config['dynamic_layers']
    dynamic_neurons = config['dynamic_neurons']
    if isinstance(dynamic_neurons, int):
        dynamic_neurons= [dynamic_neurons] * dynamic_layers
    else:
        assert len(dynamic_neurons) == dynamic_layers

    if config['n_features'] is not None:
        n_features = config['n_features']
    else:
        n_features = td.X_train_single.shape[1]

    activation_function = config['activation_function']
    if isinstance(activation_function, str):
        activation_function = [activation_function] * (encoder_layers + dynamic_layers)
    else:
        assert len(activation_function)== (encoder_layers + dynamic_layers)

    encoder_activation = activation_function[:encoder_layers]
    dynamic_activation = activation_function[encoder_layers:]

    encoder_indices = _resolve_feature_indices(config['encoder_features'], td.columns)
    dynamic_indices = _resolve_feature_indices(config['dynamic_features'], td.columns)

    prediction_loss_scale = float(np.std(td.y_train_single, ddof=1))
    #prediction_loss_scale=1.0

    t_wall_train = calculate_wall_temperature_for_scaling(
        x=td.X_train_single,
        measured_change_t_air=td.y_train_single,
        columns=td.columns,
        rc_kwargs=rc_kwargs
    )

    physics_loss_scale = float(np.std(t_wall_train, ddof=1))
    #physics_loss_scale=1.0

    # -------------------------------------------------------------------------
    # neural core model
    # -------------------------------------------------------------------------
    core_input = keras.layers.Input(shape=(n_features,), name='core_input')

    x_encoder_input = InputSliceLayer(feature_indices=encoder_indices, name='encoder_input_slice')(core_input)
    x_dynamic_input = InputSliceLayer(feature_indices=dynamic_indices, name='dynamic_input_slice')(core_input)

    # Add normalization layer
    if config['normalize']:
        encoder_normalization = keras.layers.Normalization()
        encoder_normalization.adapt(td.X_train_single[:,encoder_indices])
        x_encoder = encoder_normalization(x_encoder_input)

        dynamic_normalization = keras.layers.Normalization()
        dynamic_normalization.adapt(td.X_train_single[:,dynamic_indices])
        x_dynamic = dynamic_normalization(x_dynamic_input)
    else:
        x_encoder = x_encoder_input
        x_dynamic = x_dynamic_input

    # -------------------------------------------------------------------------
    # Encoder branch
    # -------------------------------------------------------------------------
    for i in range(0, encoder_layers):
        x_encoder = keras.layers.Dense(encoder_neurons[i], activation=encoder_activation[i])(x_encoder)

    z_latent_core = keras.layers.Dense(1, activation='linear', name='t_wall_latent_dense')(x_encoder)

    # -------------------------------------------------------------------------
    # Dynamic branch
    # -------------------------------------------------------------------------
    x_dynamic = keras.layers.Concatenate()([x_dynamic, z_latent_core])
    for i in range(0, dynamic_layers):
        # For each layer add dense
        x_dynamic = keras.layers.Dense(dynamic_neurons[i], activation=dynamic_activation[i])(x_dynamic)

    # Add output layer
    output_name = 'change_t_air_dense' if config['predict_delta'] else 't_air_dense'
    y_core = keras.layers.Dense(1, activation='linear', name=output_name)(x_dynamic)

    if config['normalize']:
        z_rescale_mean = float(np.mean(td.X_train_single[:, [t_air_index]]))
        z_rescale_sigma = float(np.std(td.X_train_single[:, [t_air_index]], ddof=1))
        z_latent_core = keras.layers.Rescaling(scale=z_rescale_sigma, offset=z_rescale_mean)(z_latent_core)

    # Add rescaling
    if config['rescale_output']:
        # Rescaling for output layer
        rescale_mean = float(np.mean(td.y_train_single))
        rescale_sigma = float(np.std(td.y_train_single, ddof=1))
        y_core = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)(y_core)

    core_model = keras.models.Model(inputs=core_input, outputs=[y_core, z_latent_core], name='core_model')

    # -------------------------------------------------------------------------
    # PINN
    # -------------------------------------------------------------------------
    pinn_input = keras.layers.Input(shape=(n_features,), name='pinn_input')

    y_nn, _ = core_model(pinn_input)

    physics_layer = RC2R2CGokhalePhysNetLayer(
        trainable_rc=config['trainable_rc'],
        use_internal_gains=config["use_internal_gains"],
        **rc_kwargs
    )
    
    model = RC2R2CGokhalePhysNetKerasModel(
        inputs=pinn_input,
        outputs=y_nn,
        core_model=core_model,
        physics_layer=physics_layer,
        physics_loss_weight=config['physics_loss_weight'],
        prediction_loss_scale=prediction_loss_scale,
        physics_loss_scale=physics_loss_scale,
        name='pinn_2r2c_gokhale'
    )

    return model


def RC2R2CGokhalePhysNetWallDynamicsConstruction(config: dict, td: TrainingDataGeneric):
    """
    
    """
    config = RC2R2CGokhalePhysNetWallDynamicsConstruction_config.model_validate(config).model_dump()

    if isinstance(config['t_air_column'], str):
        t_air_index = list(td.columns).index(config['t_air_column'])
    else:
        t_air_index = config['t_air_column']

    rc_kwargs = dict(config['rc_kwargs'])
    rc_kwargs['t_air_index'] = t_air_index
    rc_kwargs['predict_delta'] = config['predict_delta']

    encoder_layers = config['encoder_layers']
    encoder_neurons = config['encoder_neurons']
    if isinstance(encoder_neurons, int):
        encoder_neurons= [encoder_neurons] * encoder_layers
    else:
        assert len(encoder_neurons) == encoder_layers

    dynamic_layers = config['dynamic_layers']
    dynamic_neurons = config['dynamic_neurons']
    if isinstance(dynamic_neurons, int):
        dynamic_neurons= [dynamic_neurons] * dynamic_layers
    else:
        assert len(dynamic_neurons) == dynamic_layers

    if config['n_features'] is not None:
        n_features = config['n_features']
    else:
        n_features = td.X_train_single.shape[1]

    activation_function = config['activation_function']
    if isinstance(activation_function, str):
        activation_function = [activation_function] * (encoder_layers + dynamic_layers)
    else:
        assert len(activation_function)== (encoder_layers + dynamic_layers)

    encoder_activation = activation_function[:encoder_layers]
    dynamic_activation = activation_function[encoder_layers:]

    encoder_indices = _resolve_feature_indices(config['encoder_features'], td.columns)
    dynamic_indices = _resolve_feature_indices(config['dynamic_features'], td.columns)

    prediction_loss_scale = float(np.std(td.y_train_single, ddof=1))
    #prediction_loss_scale=1.0

    t_wall_train = calculate_wall_temperature_for_scaling(
        x=td.X_train_single,
        measured_change_t_air=td.y_train_single,
        columns=td.columns,
        rc_kwargs=rc_kwargs
    )

    physics_loss_scale = float(np.std(t_wall_train, ddof=1))
    #physics_loss_scale=1.0

    # -------------------------------------------------------------------------
    # neural core model
    # -------------------------------------------------------------------------
    core_input = keras.layers.Input(shape=(n_features,), name='core_input')

    x_encoder_input = InputSliceLayer(feature_indices=encoder_indices, name='encoder_input_slice')(core_input)
    x_dynamic_input = InputSliceLayer(feature_indices=dynamic_indices, name='dynamic_input_slice')(core_input)

    # Add normalization layer
    if config['normalize']:
        encoder_normalization = keras.layers.Normalization()
        encoder_normalization.adapt(td.X_train_single[:,encoder_indices])
        x_encoder = encoder_normalization(x_encoder_input)

        dynamic_normalization = keras.layers.Normalization()
        dynamic_normalization.adapt(td.X_train_single[:,dynamic_indices])
        x_dynamic = dynamic_normalization(x_dynamic_input)
    else:
        x_encoder = x_encoder_input
        x_dynamic = x_dynamic_input

    # -------------------------------------------------------------------------
    # Encoder branch
    # -------------------------------------------------------------------------
    for i in range(0, encoder_layers):
        x_encoder = keras.layers.Dense(encoder_neurons[i], activation=encoder_activation[i])(x_encoder)

    z_latent_core = keras.layers.Dense(1, activation='linear', name='t_wall_latent_dense')(x_encoder)

    # -------------------------------------------------------------------------
    # Dynamic branch
    # -------------------------------------------------------------------------
    x_dynamic = keras.layers.Concatenate()([x_dynamic, z_latent_core])
    for i in range(0, dynamic_layers):
        # For each layer add dense
        x_dynamic = keras.layers.Dense(dynamic_neurons[i], activation=dynamic_activation[i])(x_dynamic)

    # Add output layer
    output_name = 'change_t_air_dense' if config['predict_delta'] else 't_air_dense'
    y_core = keras.layers.Dense(1, activation='linear', name=output_name)(x_dynamic)

    if config['normalize']:
        z_rescale_mean = float(np.mean(td.X_train_single[:, [t_air_index]]))
        z_rescale_sigma = float(np.std(td.X_train_single[:, [t_air_index]], ddof=1))
        z_latent_core = keras.layers.Rescaling(scale=z_rescale_sigma, offset=z_rescale_mean)(z_latent_core)

    # Add rescaling
    if config['rescale_output']:
        # Rescaling for output layer
        rescale_mean = float(np.mean(td.y_train_single))
        rescale_sigma = float(np.std(td.y_train_single, ddof=1))
        y_core = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)(y_core)

    core_model = keras.models.Model(inputs=core_input, outputs=[y_core, z_latent_core], name='core_model')

    # -------------------------------------------------------------------------
    # PINN
    # -------------------------------------------------------------------------
    pinn_input = keras.layers.Input(shape=(n_features,), name='pinn_input')

    y_nn, _ = core_model(pinn_input)

    physics_layer = RC2R2CGokhalePhysNetWallDynamicsLayer(
        trainable_rc=config['trainable_rc'],
        use_internal_gains=config["use_internal_gains"],
        **rc_kwargs
    )
    
    model = RC2R2CGokhalePhysNetWallDynamicsKerasModel(
        inputs=pinn_input,
        outputs=y_nn,
        core_model=core_model,
        physics_layer=physics_layer,
        physics_loss_weight=config['physics_loss_weight'],
        wall_dynamics_loss_weight=config['wall_dynamics_loss_weight'],
        prediction_loss_scale=prediction_loss_scale,
        physics_loss_scale=physics_loss_scale,
        name='pinn_2r2c_gokhale_wall_dynamics',
    )

    return model