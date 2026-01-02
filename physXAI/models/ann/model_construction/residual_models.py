import os
import numpy as np
from sklearn.linear_model import LinearRegression
from physXAI.preprocessing.training_data import TrainingDataGeneric
from physXAI.models.ann.model_construction.rbf_models import RBFModelConstruction
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def LinResidualANNConstruction(config: dict, td: TrainingDataGeneric, lin_model: LinearRegression):
    """
        Constructs a hybrid Keras model that combines a pre-trained linear regression model
        with a Radial Basis Function (RBF) network. The RBF network is trained to model
        the residuals (errors) of the linear regression model.

        The final prediction is the sum of the linear model's prediction and the RBF network's
        prediction (which learns the residuals).

        Args:
            config (dict): A dictionary containing configuration parameters, primarily for
                           the RBF network part of the model. This config will be modified
                           to set rescaling parameters for the RBF network based on the
                           linear model's residuals.
            td (TrainingDataGeneric): An object containing the training data.
                               Used to calculate residuals and for the RBF model construction.
            lin_model (sklearn.linear_model.LinearRegression): A pre-trained scikit-learn
                                                               LinearRegression model.

        Returns:
            keras.Model: The constructed Keras functional model, which combines the linear
                         model and the residual-fitting RBF network.
    """

    # Determine predictions of linear regression for rescaling
    y_train_pred = lin_model.predict(td.X_train_single)
    config['rescale_sigma'] = float(np.std(td.y_train_single - y_train_pred, ddof=1))
    config['rescale_mean'] = 0

    # Add linear regression as dense keras layer
    lin = keras.layers.Dense(1, activation='linear')

    # Add input layer
    n_featues = td.X_train_single.shape[1]
    inputs = keras.layers.Input(shape=(n_featues,))

    # Construct rbf model
    rbf_model = RBFModelConstruction(config, td)

    # Combine linear layer and rbf model
    output = keras.layers.Add()([rbf_model(inputs), lin(inputs)])

    # Create model
    model = keras.Model(inputs, output)

    # Fix the weights of linear regression
    lin.set_weights([lin_model.coef_.reshape(-1, 1), np.array(lin_model.intercept_)])
    lin.trainable = False

    model.summary()

    return model
