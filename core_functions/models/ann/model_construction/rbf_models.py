import keras
from sklearn.cluster import KMeans
from core_functions.preprocessing.training_data import TrainingDataGeneric
from core_functions.models.ann.configs.ann_model_configs import RBFConstruction_config
from core_functions.models.ann.keras_models.keras_models import RBFLayer


def RBFModelConstruction(config: dict, td: TrainingDataGeneric):
    """
    Constructs a Radial Basis Function (RBF) Network model using Keras.

    The first RBF layer's centers can be initialized using K-Means clustering
    on the training data. Subsequent RBF layers (if any) will have their
    centers initialized by the RBFLayer's default mechanism or as specified.

    Args:
        config (dict): A dictionary containing the configuration parameters for the RBF network.
                       This is validated against `ClassicalANNConstruction_config`.
        td (TrainingDataGeneric): An object containing the training data,
                           used for adapting normalization, K-Means clustering, and
                           determining input/output shapes.

    Returns:
        keras.Model: The constructed Keras functional model representing the RBF network.
    """

    # Validate the input configuration dictionary using the Pydantic model and convert it to a dictionary
    config = RBFConstruction_config.model_validate(config).model_dump()

    # Get config
    n_layers = config['n_layers']
    n_neurons = config['n_neurons']
    # If n_neurons is a single integer, replicate it for all layers
    if isinstance(n_neurons, int):
        n_neurons = [n_neurons] * n_layers
    else:
        assert len(n_neurons) == n_layers
    n_featues = td.X_train_single.shape[1]

    # Rescaling for output layer
    # Custom rescaling
    if 'rescale_scale' in config.keys() and config['rescale_scale'] is not None:
        if 'rescale_offset' in config.keys() and config['rescale_offset'] is not None:
            offset = config['rescale_offset']
        else:
            offset = 0
        rescale_scale = config['rescale_scale']
        rescale_min = offset
        rescale_max = offset + rescale_scale
    # Standard rescaling
    else:
        rescale_min = float(td.y_train_single.min())
        rescale_max = float(td.y_train_single.max())

    # Add input layer
    input_layer = keras.layers.Input(shape=(n_featues,))
    # Add normalization layer
    normalization = keras.layers.Normalization()
    normalization.adapt(td.X_train_single)
    x = normalization(input_layer)

    for i in range(0, n_layers):
        # For each layer add RBF

        # Determine initial rbf centers
        if i == 0:
            # Apply KMeans Clustering for rbf centers
            kmeans = KMeans(n_clusters=n_neurons[i], random_state=config['random_state'], n_init='auto')
            kmeans.fit(normalization(td.X_train_single))
            initial_centers_kmeans = kmeans.cluster_centers_
            x = RBFLayer(n_neurons[i], initial_centers=initial_centers_kmeans, gamma=1)(x)
        else:
            x = RBFLayer(n_neurons[i], gamma=1)(x)

    # Add output layer
    x = keras.layers.Dense(1, activation='linear')(x)

    # Add rescaling
    if config['rescale_output']:
        x = keras.layers.Rescaling(scale=rescale_max - rescale_min, offset=rescale_min)(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    model.summary()

    return model
