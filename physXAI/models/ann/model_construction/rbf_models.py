import keras
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from physXAI.preprocessing.training_data import TrainingDataGeneric
from physXAI.models.ann.configs.ann_model_configs import RBFConstruction_config
from physXAI.models.ann.keras_models.keras_models import RBFLayer


def gamma_init(centers, overlap=0.5) -> float:
    """Initialize gamma parameter for RBF layer based on centers (median nearest-neighbor distance) and desired overlap.

    Args:
        centers (np.ndarray): Array of shape (n_centers, n_features) representing the RBF centers.
        overlap (float): Desired overlap factor between RBFs. Higher values lead to more overlap.

    Returns:
        gamma: Calculated gamma value for the RBF layer (gamma = -np.log(overlap) / avg_dist_sq).
    """
    nbrs = NearestNeighbors(n_neighbors=2).fit(centers)
    distances, _ = nbrs.kneighbors(centers)
    dist_sq = distances[:, 1] ** 2
    avg_dist_sq = np.median(dist_sq)

    if avg_dist_sq == 0:
        return 1.0 # Fallback
    
    gamma = -np.log(overlap) / avg_dist_sq
    # print(f"Calculated Gamma: {gamma}")
    return gamma
    


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
    n_neurons = config['n_neurons']
    # If n_neurons is a single integer, replicate it for all layers
    if isinstance(n_neurons, list):
        n_neurons = n_neurons[0]
    if config['n_features'] is not None:
        n_features = config['n_features']
    else:
        n_features = td.X_train_single.shape[1]

    # Add input layer
    input_layer = keras.layers.Input(shape=(n_features,))
    # Add normalization layer
    if config['normalize']:
        normalization = keras.layers.Normalization()
        normalization.adapt(td.X_train_single)
        x = normalization(input_layer)
    else:
        x = input_layer

    kmeans = KMeans(n_clusters=n_neurons, random_state=config['random_state'], n_init='auto')
    kmeans.fit(normalization(td.X_train_single).numpy())
    initial_centers_kmeans = kmeans.cluster_centers_
    
    x = RBFLayer(n_neurons, 
                 initial_centers=initial_centers_kmeans, 
                 gamma=gamma_init(initial_centers_kmeans, overlap=0.5),
                 learnable_centers=False,
                 learnable_gamma=False)(x)

    # Add output layer
    x = keras.layers.Dense(1, activation='linear', use_bias=False)(x)

    # Add rescaling
    if config['rescale_output']:

        # Rescaling for output layer
        # Custom rescaling
        # --- Sigma (Scale) ---
        if 'rescale_sigma' in config and config['rescale_sigma'] is not None:
            rescale_sigma = config['rescale_sigma']
        else:
            # Auto-calculate from data
            rescale_sigma = float(np.std(td.y_train_single, ddof=1))
        # --- Mean (Offset) ---
        # CASE A: Residual Mode -> Config must provide 0.0
        # CASE B: Direct Prediction -> Config is None, calculate from data
        if 'rescale_mean' in config and config['rescale_mean'] is not None:
            rescale_mean = config['rescale_mean']
        else:
            rescale_mean = float(np.mean(td.y_train_single))

        x = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    model.summary()

    return model
