import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from physXAI.utils.logging import create_full_path, get_full_path, Logger
from physXAI.preprocessing.training_data import TrainingData, TrainingDataMultiStep, TrainingDataGeneric
from physXAI.models.models import SingleStepModel, LinearRegressionModel, MultiStepModel, register_model
from physXAI.models.ann.model_construction.ann_models import ClassicalANNConstruction, CMNNModelConstruction
from physXAI.models.ann.model_construction.rbf_models import RBFModelConstruction
from physXAI.models.ann.model_construction.redidual_models import LinResidualANNConstruction
from physXAI.models.ann.model_construction.rnn_models import RNNModelConstruction
from physXAI.models.ann.pinn.pinn_loss import multi_y_loss
from physXAI.plotting.plotting import plot_prediction_correlation, plot_predictions, plot_training_history, \
    plot_metrics_table, subplots, plot_multi_rmse
from physXAI.evaluation.metrics import MetricsPINN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class ANNModel(SingleStepModel, ABC):
    """
    Abstract Base Class for single-step Artificial Neural Network models.
    Provides common functionality for compiling, fitting, plotting, saving,
    loading, and managing configurations for Keras-based ANN models.
    """

    def __init__(self, batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: Optional[int] = 100, random_seed: int = 42, **kwargs):
        """
        Initializes common hyperparameters for ANN training.

        Args:
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                         If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility.
        """
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.early_stopping_epochs: Optional[int] = early_stopping_epochs
        self.random_seed: int = random_seed
        keras.utils.set_random_seed(random_seed)

    @abstractmethod
    def generate_model(self, **kwargs):
        """
        Abstract method to be implemented by subclasses to define and return a Keras model.
        The `td` (TrainingData) object is expected to be passed via `kwargs`.
        """
        return None

    def compile_model(self, model):
        """
        Compiles the Keras model with Adam optimizer, Mean Squared Error loss,
        and Root Mean Squared Error metric.

        Args:
            model (keras.Model): The Keras model to compile.
        """
        model.compile(keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse',
                      metrics=[keras.metrics.RootMeanSquaredError(name='rmse', dtype=None)])

    def fit_model(self, model, td: TrainingDataGeneric):
        """
         Fits the Keras model to the training data.

         Args:
             model (keras.Model): The Keras model to fit.
             td (TrainingDataGeneric): The TrainingData object
         """

        # Early stopping
        callbacks = list()
        if self.early_stopping_epochs is not None:
            es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.early_stopping_epochs,
                                               restore_best_weights=True, verbose=1)
            callbacks.append(es)

        # Check for validation data
        if td.y_val is not None:
            val_data = (td.X_val_single, td.y_val_single)
        else:
            val_data = None

        # Fit model, track training time
        start_time = time.perf_counter()
        training_history = model.fit(td.X_train_single, td.y_train_single,
                                     validation_data=val_data,
                                     batch_size=self.batch_size, epochs=self.epochs,
                                     callbacks=callbacks)
        stop_time = time.perf_counter()

        # Add metrics to training data
        td.add_training_time(stop_time - start_time)
        td.add_training_record(training_history)

        model.summary()

    def plot(self, td: TrainingDataGeneric):
        """
        Generates and displays various plots related to model performance and training.

        Args:
            td (TrainingDataGeneric): The TrainingData object
        """

        fig1 = plot_prediction_correlation(td)
        fig2 = plot_predictions(td)
        fig3 = plot_training_history(td)
        fig4 = plot_metrics_table(td)

        # Create main plot
        if isinstance(td, TrainingData):
            subplots(
                "Artificial Neural Network",
                {"title": "Prediction Correlation", "type": "scatter", "figure": fig1},
                {"title": "Predictions Sorted", "type": "scatter", "figure": fig2},
                {"title": "Training History", "type": "scatter", "figure": fig3},
                {"title": "Performance Metrics", "type": "table", "figure": fig4}
            )
        elif isinstance(td, TrainingDataMultiStep):
            fig5 = plot_multi_rmse(td)
            subplots(
                "Artificial Neural Network",
                # {"title": "Prediction Correlation", "type": "scatter", "figure": fig1},
                {"title": "Predictions Sorted", "type": "scatter", "figure": fig2},
                {"title": "Prediction Step RMSE", "type": "scatter", "figure": fig5},
                {"title": "Training History", "type": "scatter", "figure": fig3},
                {"title": "Performance Metrics", "type": "table", "figure": fig4}
            )
        else:
            raise NotImplementedError

    def save_model(self, model, save_path: str):
        """
        Saves the Keras model to the specified path.

        Args:
            model (keras.Model): The Keras model to save.
            save_path (str): The directory or full path where the model should be saved.
        """

        if save_path is None:
            save_path = Logger.get_model_savepath()

        if not save_path.endswith('.keras'):
            save_path += '.keras'

        save_path = create_full_path(save_path)
        model.save(save_path)

    def load_model(self, load_path: str):
        """
        Loads a Keras model from the specified path.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            keras.Model: The loaded Keras model.
        """

        load_path = get_full_path(load_path)
        model = keras.saving.load_model(load_path)
        return model

    def get_config(self) -> dict:
        c = super().get_config()
        c.update({
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'early_stopping_epochs': self.early_stopping_epochs,
            'random_seed': self.random_seed
        })
        return c


@register_model
class ClassicalANNModel(ANNModel):
    """
    A classical (standard feed-forward) Artificial Neural Network model.
    """

    def __init__(self, n_layers: int = 1, n_neurons: int or list[int] = 32,
                 activation_function: str or list[str] = 'softplus', rescale_output: bool = True,
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: Optional[int] = 100, random_seed: int = 42, **kwargs):
        """
          Initializes the ClassicalANNModel.

          Args:
              n_layers (int): Number of hidden layers.
              n_neurons (int or list[int]): Number of neurons in each hidden layer.
                                            If int, same for all. If list, specifies for each layer.
              activation_function (str or list[str]): Activation function(s) for hidden layers.
                                                      If str, same for all. If list, specifies for each layer.
              rescale_output (bool): Whether to rescale the model's output to the original target range.
              batch_size (int): Number of samples per gradient update.
              epochs (int): Number of times to iterate over the entire training dataset.
              learning_rate (float): Learning rate for the Adam optimizer.
              early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                           If None, early stopping is disabled.
              random_seed (int): Seed for random number generators to ensure reproducibility.
        """

        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed)
        self.n_layers: int = n_layers
        self.n_neurons: int or list[int] = n_neurons
        self.activation_function: str or list[str] = activation_function
        self.rescale_output: bool = rescale_output

        self.model_config = {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
        }

    def generate_model(self, **kwargs):
        """
        Generates the Keras model using ClassicalANNConstruction.
        """

        td = kwargs['td']
        model = ClassicalANNConstruction(self.model_config, td)
        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output
        })
        return config


@register_model
class LinANNModel(ANNModel):
    """
    A hybrid model combining a Linear Regression model with an ANN (likely RBF)
    that models the residuals of the linear regression.
    """

    def __init__(self, n_layers: int = 1, n_neurons: int or list[int] = 32, rescale_output: bool = True,
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int = 100, random_seed: int = 42, **kwargs):
        """
        Initializes the LinANNModel.

        Args:
            n_layers (int): Number of hidden layers for the residual-fitting ANN.
            n_neurons (int or list[int]): Number of neurons for the residual-fitting ANN.
            rescale_output (bool): Whether to rescale the final combined output.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                           If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility.
        """
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed)
        self.n_layers: int = n_layers
        self.n_neurons: int or list[int] = n_neurons
        self.rescale_output: bool = rescale_output

        self.model_config = {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output,
            "random_state": random_seed
        }

    def generate_model(self, **kwargs):
        """
        Generates the hybrid Linear + ANN model.
        First, a Linear Regression model is trained. Then, an ANN (e.g., RBF)
        is constructed to model its residuals.
        """

        td = kwargs['td']

        # Train linear regression
        lr = LinearRegressionModel()
        lr_model = lr.generate_model()
        lr.fit_model(lr_model, td)
        lr.evaluate(lr_model, td)

        # Construct residual model
        model = LinResidualANNConstruction(self.model_config, td, lr_model)
        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output
        })
        return config


@register_model
class RBFModel(ANNModel):
    """
    A Radial Basis Function (RBF) Network model.
    """

    def __init__(self, n_layers: int = 1, n_neurons: int or list[int] = 32, rescale_output: bool = True,
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int = 100, random_seed: int = 42, **kwargs):
        """
        Initializes the RBFModel.

        Args:
            n_layers (int): Number of RBF layers.
            n_neurons (int or list[int]): Number of RBF neurons in each layer.
            rescale_output (bool): Whether to rescale the model's output.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                           If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility.
        """
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed)
        self.n_layers: int = n_layers
        self.n_neurons: int or list[int] = n_neurons
        self.rescale_output: bool = rescale_output

        self.model_config = {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output,
            "random_state": random_seed
        }

    def generate_model(self, **kwargs):
        """
        Generates the Keras RBF model using RBFModelConstruction.
        """

        td = kwargs['td']
        model = RBFModelConstruction(self.model_config, td)
        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output
        })
        return config


@register_model
class CMNNModel(ANNModel):
    """
    A Constrained Monotonic Neural Network (CMNN) model.
    Allows enforcing monotonicity constraints on input features.
    """

    def __init__(self, n_layers: int = 1, n_neurons: int or list[int] = 32,
                 activation_function: str or list[str] = 'softplus', rescale_output: bool = True,
                 monotonies: dict[str, int] = None, activation_split: list[float] = None,
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int = 100, random_seed: int = 42, **kwargs):
        """
        Initializes the CMNNModel.

        Args:
            n_layers (int): Number of hidden layers.
            n_neurons (int or list[int]): Number of neurons per layer.
            activation_function (str or list[str]): Activation function(s).
            rescale_output (bool): Whether to rescale output.
            monotonies (dict[str, int]): Dictionary mapping feature names to monotonicity type
                                         (-1 for decreasing, 0 for no constraint, 1 for increasing).
            activation_split (list[float]): Proportions for splitting neurons into convex,
                                            concave, and saturated activation paths. E.g., [0.5, 0.25, 0.25].
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                           If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility.
        """
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed)
        self.n_layers: int = n_layers
        self.n_neurons: int or list[int] = n_neurons
        self.activation_function: str or list[str] = activation_function
        self.rescale_output: bool = rescale_output
        self.monotonies: dict[str, int] = monotonies
        self.activation_split: list[float] = activation_split

        self.model_config = {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
            "monotonicities": self.monotonies,
            "activation_split": activation_split,
        }

    def generate_model(self, **kwargs):
        """
        Generates the Keras CMNN model using CMNNModelConstruction.
        """

        td = kwargs['td']
        model = CMNNModelConstruction(self.model_config, td)
        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
            "monotonies": self.monotonies,
            "activation_split": self.activation_split
        })
        return config


@register_model
class PINNModel(ANNModel):
    """
    A Physics-Informed Neural Network (PINN) model.
    This implementation uses a CMNN as its base architecture and incorporates
    a custom multi-component loss function.
    """

    def __init__(self, n_layers: int = 1, n_neurons: int or list[int] = 32,
                 activation_function: str or list[str] = 'softplus', pinn_weights: list[float] = None,
                 rescale_output: bool = True, monotonies: dict[str, int] = None, activation_split: list[float] = None,
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int = 100, random_seed: int = 42, **kwargs):
        """
        Initializes the PINNModel.

        Args:
            pinn_weights (list[float]): Weights for the additional components in the multi_y_loss.
                                       The length should be `num_target_components - 1`.
            n_layers (int): Number of hidden layers.
            n_neurons (int or list[int]): Number of neurons per layer.
            activation_function (str or list[str]): Activation function(s).
            rescale_output (bool): Whether to rescale output.
            monotonies (dict[str, int]): Dictionary mapping feature names to monotonicity type
                                         (-1 for decreasing, 0 for no constraint, 1 for increasing).
            activation_split (list[float]): Proportions for splitting neurons into convex,
                                            concave, and saturated activation paths. E.g., [0.5, 0.25, 0.25].
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                           If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility.
        """
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed)
        self.n_layers: int = n_layers
        self.n_neurons: int or list[int] = n_neurons
        self.activation_function: str or list[str] = activation_function
        self.rescale_output: bool = rescale_output
        self.monotonies: dict[str, int] = monotonies
        self.activation_split: list[float] = activation_split

        self.pinn_weights: list[float] = pinn_weights

        self.model_config = {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
            "monotonicities": self.monotonies,
            "activation_split": activation_split,
        }

        # Create pinn loss based on standard losses
        self.pinn_loss = multi_y_loss(keras.losses.MeanSquaredError(name='MSE'), self.pinn_weights, 'mse')
        self.pinn_metrics = [multi_y_loss(keras.metrics.RootMeanSquaredError(name='rmse'), self.pinn_weights,
                                          'rmse')]

    def generate_model(self, **kwargs):
        """
        Generates the Keras model (typically a CMNN) to be used as the PINN.
        """

        td = kwargs['td']
        model = CMNNModelConstruction(self.model_config, td)
        return model

    def compile_model(self, model):
        """
        Compiles the PINN model with the custom multi_y_loss and corresponding metrics.
        """

        model.compile(keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=self.pinn_loss,
                      metrics=self.pinn_metrics)

    def _check_pinn_weights(self, td: TrainingDataGeneric):
        """
        Checks and sets default PINN weights if not provided, based on the shape of y_train.
        y_train is expected to have multiple components: y_train[:, 0] for data-driven loss,
        and y_train[:, 1:] for physics-informed/residual losses.

        Args:
            td (TrainingDataGeneric): The training data, used to infer the number of target components.
        """

        yn = td.y_train_single.shape[1]
        if self.pinn_weights is None:
            if yn == 1:  # pragma: no cover
                self.pinn_weights = list()  # pragma: no cover
            else:
                self.pinn_weights = [1] * (yn - 1)
            self.pinn_loss = multi_y_loss(keras.losses.MeanSquaredError(name='MSE'), self.pinn_weights, 'mse')
            self.pinn_metrics = [multi_y_loss(keras.metrics.RootMeanSquaredError(name='rmse'), self.pinn_weights,
                                              'rmse')]
        assert yn == len(self.pinn_weights) + 1, \
            f'Shape of y values does not match length of pinn weights'

    def pipeline(self, td: TrainingDataGeneric,
                 save_path: str = None, plot: bool = True, save_model: bool = True):
        """
        Overrides the base pipeline to include PINN weight checking.

        Args:
            td (TrainingData): The training data, used to infer the number of target components.
            save_path (str): Path to save the model.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            save_model (bool, optional): Whether to save the trained model. Defaults to True.
        """

        self._check_pinn_weights(td)
        super().pipeline(td, save_path, plot, save_model)

    def online_pipeline(self, td: TrainingDataGeneric, load_path: str = None, save_path: str = None,
                        plot: bool = True, save_model: bool = True):
        """
        Overrides the online pipeline to include PINN weight checking.

        Args:
            td (TrainingData): The training data, used to infer the number of target components.
            load_path (str): Path to load the model.
            save_path (str): Path to save the model (If None, standard save path of Logger is used).
            plot (bool, optional): Whether to plot the results. Defaults to True.
            save_model (bool, optional): Whether to save the trained model. Defaults to True.
        """
        self._check_pinn_weights(td)
        super().online_pipeline(td, load_path, save_path, plot, save_model)

    def evaluate(self, model, td: TrainingDataGeneric):
        """
        Evaluates the PINN model using custom MetricsPINN.

        Args:
            model: The keras model to be evaluated.
            td (TrainingData): The training data
        """

        y_pred_train = model.predict(td.X_train_single)
        y_pred_test = model.predict(td.X_test_single)
        if td.X_val is not None:
            y_pred_val = model.predict(td.X_val_single)
        else:
            y_pred_val = None
        td.add_predictions(y_pred_train, y_pred_val, y_pred_test)

        metrics = MetricsPINN(td, [self.pinn_loss, *self.pinn_metrics])
        td.add_metrics(metrics)

    def load_model(self, load_path: str):
        """
        Loads a Keras model from the specified path.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            keras.Model: The loaded Keras model.
        """

        load_path = get_full_path(load_path)
        model = keras.saving.load_model(load_path, compile=False)
        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "pinn_weights": self.pinn_weights,
            "rescale_output": self.rescale_output,
            "monotonies": self.monotonies,
            "activation_split": self.activation_split
        })
        return config


@register_model
class RNNModel(MultiStepModel):
    """
    A Recurrent Neural Network (RNN) model for multi-step forecasting.
    Inherits from MultiStepModel.
    """

    def __init__(self, rnn_units: int = 32, rnn_layer: str = 'RNN', init_layer=None, epochs: int = 1000,
                 learning_rate: float = 0.001, early_stopping_epochs: Optional[int] = 100, random_seed: int = 42, **kwargs):
        """
        Initializes the RNNModel.

        Args:
            rnn_units (int): Number of units in the RNN layer.
            rnn_layer (str): Type of RNN layer ('RNN', 'LSTM', 'GRU').
            init_layer (str, optional): Type of layer  ('dense', 'RNN', 'LSTM', 'GRU')
                                        used for initializing RNN state if warmup is used.
                                        Defaults to the same as `rnn_layer`.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                         If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility.
        """

        super().__init__(**kwargs)

        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.early_stopping_epochs: int = early_stopping_epochs
        self.random_seed: int = random_seed
        keras.utils.set_random_seed(random_seed)

        if init_layer is None:
            init_layer = rnn_layer

        self.rnn_units: int = rnn_units
        self.rnn_layer: str = rnn_layer
        self.init_layer: str = init_layer

        self.model_config = {
            'rnn_units': rnn_units,
            'init_layer': init_layer,
            'rnn_layer': rnn_layer,
        }

    def generate_model(self, **kwargs):
        """
        Generates the Keras RNN model using RNNModelConstruction.
        """

        td = kwargs['td']
        model = RNNModelConstruction(self.model_config, td)
        return model

    def compile_model(self, model):
        """
        Compiles the RNN model.
        """

        model.compile(keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse',
                      metrics=[keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)])

    def fit_model(self, model, td: TrainingDataMultiStep):
        """
         Fits the Keras model to the training data.

         Args:
             model (keras.Model): The Keras model to fit.
             td (TrainingDataMultiStep): The TrainingData object from TrainingDataMultiStep
         """

        # Early stopping
        callbacks = list()
        if self.early_stopping_epochs is not None:
            es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.early_stopping_epochs,
                                               restore_best_weights=True, verbose=1)
            callbacks.append(es)

        # Fit model, track training time
        start_time = time.perf_counter()
        training_history = model.fit(td.train_ds, validation_data=td.val_ds, epochs=self.epochs, callbacks=callbacks)
        stop_time = time.perf_counter()

        # Add metrics to training data
        td.add_training_time(stop_time - start_time)
        td.add_training_record(training_history)

    def plot(self, td: TrainingDataMultiStep):
        """
        Generates and displays various plots related to model performance and training.

        Args:
            td (TrainingDataMultiStep): The TrainingData object from TrainingDataMultiStep
        """

        # fig1 = plot_prediction_correlation(td)
        fig2 = plot_predictions(td)
        fig3 = plot_training_history(td)
        fig4 = plot_metrics_table(td)
        fig5 = plot_multi_rmse(td)

        subplots(
            "Recurrent Neural Network",
            # {"title": "Prediction Correlation", "type": "scatter", "figure": fig1},
            {"title": "Predictions Sorted", "type": "scatter", "figure": fig2},
            {"title": "Prediction Step RMSE", "type": "scatter", "figure": fig5},
            {"title": "Training History", "type": "scatter", "figure": fig3},
            {"title": "Performance Metrics", "type": "table", "figure": fig4}
        )

    def save_model(self, model, save_path: str):
        """
        Saves the Keras model to the specified path.

        Args:
            model (keras.Model): The Keras model to save.
            save_path (str): The directory or full path where the model should be saved.
        """

        if save_path is None:
            save_path = Logger.get_model_savepath()

        if not save_path.endswith('.keras'):
            save_path += '.keras'

        save_path = create_full_path(save_path)
        model.save(save_path)

    def load_model(self, load_path: str):
        """
        Loads a Keras model from the specified path.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            keras.Model: The loaded Keras model.
        """

        load_path = get_full_path(load_path)
        model = keras.saving.load_model(load_path)
        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'rnn_units': self.rnn_units,
            'rnn_layer': self.rnn_layer,
            'init_layer': self.init_layer,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'early_stopping_epochs': self.early_stopping_epochs,
            'random_seed': self.random_seed
        })
        return config
