import logging
import os
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
import tensorflow as tf

from physXAI.evaluation.metrics import MetricsPINN
from physXAI.expr import Expr
from physXAI.expr import from_config as expr_from_config
from physXAI.expr import to_config as expr_to_config
from physXAI.io.run import input_path, keras_verbosity, keras_verbosity_int, output_path
from physXAI.models.ann.model_construction.ann_models import ClassicalANNConstruction, CMNNModelConstruction
from physXAI.models.ann.model_construction.rbf_models import RBFModelConstruction
from physXAI.models.ann.model_construction.residual_models import LinResidualANNConstruction
from physXAI.models.ann.model_construction.rnn_models import (
    SequenceSpec,
    build_sequence_model,
    build_step_model,
    find_step_model,
    sequence_inputs,
)
from physXAI.models.ann.pinn.pinn_loss import multi_y_loss
from physXAI.models.models import LinearRegressionModel, MultiStepModel, SingleStepModel, register_model
from physXAI.preprocessing.training_data import TrainingData, TrainingDataGeneric, TrainingDataMultiStep
from physXAI.training.recursive import step_weights, weighted_step_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

if TYPE_CHECKING:  # pragma: no cover
    from physXAI.training import RecursiveTraining

logger = logging.getLogger("physXAI")


class ANNModel(SingleStepModel, ABC):
    """
    Abstract Base Class for single-step Artificial Neural Network models.
    Provides common functionality for compiling, fitting, plotting, saving,
    loading, and managing configurations for Keras-based ANN models.
    """

    plot_title = "Artificial Neural Network"

    def __init__(self, batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int | None = 100, random_seed: int = 42,
                 recursive_training: Union['RecursiveTraining', dict] | None = None, **kwargs):
        """
        Initializes common hyperparameters for ANN training.

        Args:
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                         If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility. It is set when
                               the model is generated, not when the object is created.
            recursive_training (RecursiveTraining, optional): Train on the multi step (recursive) error,
                               see `physXAI.training.RecursiveTraining`. None trains on the single step error.
        """
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.early_stopping_epochs: int | None = early_stopping_epochs
        self.random_seed: int = random_seed
        from physXAI.training.recursive import RecursiveTraining
        if isinstance(recursive_training, dict):
            recursive_training = RecursiveTraining.from_config(recursive_training)
        self.recursive_training: RecursiveTraining | None = recursive_training

        self.model_config = dict()


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

    def prepare(self):
        """Seeds keras before the model is generated, so weight initializations are reproducible."""
        keras.utils.set_random_seed(self.random_seed)

    def fit_model(self, model, td: TrainingDataGeneric):
        """
        Fits the Keras model to the training data: on the single step error and / or, with
        `recursive_training`, on the multi step error of a recursive application.

        Args:
            model (keras.Model): The Keras model to fit.
            td (TrainingDataGeneric): The TrainingData object
        """
        mode = self.recursive_training.mode if self.recursive_training is not None else 'single_step'
        if mode in ('single_step', 'single_then_multi'):
            self.fit_single_step(model, td)
        if mode in ('multi_step', 'single_then_multi'):
            self.fit_multi_step(model, td)

    def fit_multi_step(self, model, td: TrainingDataGeneric):
        """Trains the model on the multi step error (`physXAI.training.recursive`)."""
        from physXAI.training.recursive import fit_recursive
        if not isinstance(td, TrainingData):
            raise NotImplementedError('Multi step training of single step models requires PreprocessingSingleStep data.')
        start_time = time.perf_counter()
        record = fit_recursive(model, td, self.recursive_training, self)
        td.add_training_time((td.training_time or 0) + time.perf_counter() - start_time)
        td.recursive_training_record = record
        if td.training_record is None:
            td.add_training_record(record)

    def fit_single_step(self, model, td: TrainingDataGeneric):
        """Fits the Keras model on the single step error."""
        # Early stopping (only possible with validation data)
        callbacks = list()
        if self.early_stopping_epochs is not None and td.y_val_single is not None:
            es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.early_stopping_epochs,
                                               restore_best_weights=True, verbose=keras_verbosity_int())
            callbacks.append(es)

        # Fit model, track training time
        start_time = time.perf_counter()

        # Create tf.data.Dataset for better performance
        train_ds = tf.data.Dataset.from_tensor_slices((td.X_train_single, td.y_train_single))
        train_ds = (
            train_ds
            .cache()                                                                                       # 1. Cache first so data is not reloaded, may need to be removed for large datasets
            .shuffle(buffer_size=min(10000, td.X_train_single.shape[0]), reshuffle_each_iteration=True)    # 2. Shuffle randomly every epoch
            .batch(self.batch_size)                                                                        # 3. Group into batches
            .prefetch(buffer_size=tf.data.AUTOTUNE)                                                        # 4. Prepare next batch in background
        )

        # Check for validation data
        if td.y_val is not None:
            val_ds = tf.data.Dataset.from_tensor_slices((td.X_val_single, td.y_val_single))
            val_ds = (
                val_ds
                .cache()                                        # 1. Cache first so data is not reloaded, may need to be removed for large datasets
                .batch(self.batch_size)                         # 2. Group into batches
                .prefetch(buffer_size=tf.data.AUTOTUNE)         # 3. Prepare next batch in background
            )
        else:
            val_ds = None

        training_history = model.fit(train_ds,
                                     validation_data=val_ds,
                                     epochs=self.epochs,
                                     callbacks=callbacks,
                                     verbose=keras_verbosity(),
                                     shuffle=False)  # the dataset is reshuffled in every epoch
        stop_time = time.perf_counter()

        # Add metrics to training data
        td.add_training_time(stop_time - start_time)
        td.add_training_record(training_history)

        if logger.isEnabledFor(logging.INFO):
            model.summary()

    def save_model(self, model, save_path: str):
        """
        Saves the Keras model to the specified path.

        Args:
            model (keras.Model): The Keras model to save.
            save_path (str): The directory or full path where the model should be saved.
        """
        save_path = output_path(save_path, '.keras')
        model.save(save_path)

    def load_model(self, load_path: str):
        """
        Loads a Keras model from the specified path.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            keras.Model: The loaded Keras model.
        """

        load_path = input_path(load_path)
        model = keras.saving.load_model(load_path)
        return model

    def get_config(self) -> dict:
        c = super().get_config()
        c.update({
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'early_stopping_epochs': self.early_stopping_epochs,
            'random_seed': self.random_seed,
            'recursive_training': self.recursive_training.get_config() if self.recursive_training is not None else None,
        })
        return c


@register_model
class ClassicalANNModel(ANNModel):
    """
    A classical (standard feed-forward) Artificial Neural Network model.
    """

    def __init__(self, n_layers: int = 1, n_neurons: int | list[int] = 32,
                 activation_function: str | list[str] = 'softplus', rescale_output: bool = True,
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int | None = 100, random_seed: int = 42, **kwargs):
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

        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.activation_function: str | list[str] = activation_function
        self.rescale_output: bool = rescale_output

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
        })

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

    def __init__(self, n_layers: int = 1, n_neurons: int | list[int] = 32, rescale_output: bool = True,
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
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.rescale_output: bool = rescale_output

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output,
            "random_state": random_seed
        })

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

    def __init__(self, n_layers: int = 1, n_neurons: int | list[int] = 32, rescale_output: bool = True,
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
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.rescale_output: bool = rescale_output

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output,
            "random_state": random_seed
        })

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

    def __init__(self, n_layers: int = 1, n_neurons: int | list[int] = 32,
                 activation_function: str | list[str] = 'softplus', rescale_output: bool = True,
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
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.activation_function: str | list[str] = activation_function
        self.rescale_output: bool = rescale_output
        self.monotonies: dict[str, int] = monotonies
        self.activation_split: list[float] = activation_split

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
            "monotonicities": self.monotonies,
            "activation_split": activation_split,
        })

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

    def __init__(self, n_layers: int = 1, n_neurons: int | list[int] = 32,
                 activation_function: str | list[str] = 'softplus', pinn_weights: list[float] = None,
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
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.activation_function: str | list[str] = activation_function
        self.rescale_output: bool = rescale_output
        self.monotonies: dict[str, int] = monotonies
        self.activation_split: list[float] = activation_split

        self.pinn_weights: list[float] = pinn_weights

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
            "monotonicities": self.monotonies,
            "activation_split": activation_split,
        })

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
            'Shape of y values does not match length of pinn weights'

    def pipeline(self, td: TrainingDataGeneric,
                 save_path: str = None, plot: bool = True, save_model: bool = True, show_plot: bool = False):
        """
        Overrides the base pipeline to include PINN weight checking.

        Args:
            td (TrainingData): The training data, used to infer the number of target components.
            save_path (str): Path to save the model.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            save_model (bool, optional): Whether to save the trained model. Defaults to True.
        """

        self._check_pinn_weights(td)
        return super().pipeline(td, save_path, plot, save_model, show_plot)

    def online_pipeline(self, td: TrainingDataGeneric, load_path: str = None, save_path: str = None,
                        plot: bool = True, save_model: bool = True, show_plot: bool = False):
        """
        Overrides the online pipeline to include PINN weight checking.

        Args:
            td (TrainingData): The training data, used to infer the number of target components.
            load_path (str): Path to load the model.
            save_path (str): Path to save the model (not saved without a path).
            plot (bool, optional): Whether to plot the results. Defaults to True.
            save_model (bool, optional): Whether to save the trained model. Defaults to True.
        """
        self._check_pinn_weights(td)
        return super().online_pipeline(td, load_path, save_path, plot, save_model, show_plot)

    def evaluate(self, model, td: TrainingDataGeneric):
        """
        Evaluates the PINN model using custom MetricsPINN.

        Args:
            model: The keras model to be evaluated.
            td (TrainingData): The training data
        """

        y_pred_train = model.predict(td.X_train_single, verbose=keras_verbosity())
        y_pred_test = model.predict(td.X_test_single, verbose=keras_verbosity())
        if td.X_val is not None:
            y_pred_val = model.predict(td.X_val_single, verbose=keras_verbosity())
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

        load_path = input_path(load_path)
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
    Recurrent neural network (SimpleRNN, GRU or LSTM) for multi step predictions.

    The model warms itself up: starting from zero states, it is applied to the `warmup_width`
    measured steps in front of the prediction, which are not part of the loss. AgentLib-MPC warms the
    model up the same way, so there is no separate initialization model.

    Predicting the change of a feature (output `px.Change('T')`) is recommended. The trajectory of T
    is then the measured start value plus the accumulated changes, and the loss is computed on that
    trajectory (`loss_on='trajectory'`). If T is also an input of the model, the predicted T is fed
    back during the prediction (closed loop), as in an MPC.

    The saved model (`save_model`, `save_for_agentlib`) is the step model
    `[sequence, *states] -> [prediction, *states]`, which AgentLib-MPC evaluates one step at a time.

    Args:
        rnn_units: Number of units of the recurrent layer.
        rnn_layer: Type of the recurrent layer ('RNN', 'GRU' or 'LSTM').
        epochs: Number of training epochs.
        learning_rate: Learning rate of the Adam optimizer.
        batch_size: Number of sequences per batch.
        early_stopping_epochs: Patience of the early stopping on the validation loss (None disables it).
        random_seed: Seed of the weight initialization, set when the model is generated.
        feature_architecture: An expression (`physXAI.expr`), or a list of them, building the inputs of
                              the recurrent layer out of the input features. Variables are not
                              normalized, wrap them in `px.Normalize` where needed.
        loss_on: 'trajectory' (default): loss on the trajectory of the predicted features, i.e. on the
                 accumulated changes. 'outputs': loss on the outputs (e.g. the changes) themselves.
        first_step_weight: Optional share of the first predicted step in the loss, the other steps share
                           the rest. None weights all steps equally.
        step_weights: Optional weight per predicted step (overrides `first_step_weight`).
    """

    plot_title = "Recurrent Neural Network"

    def __init__(
        self,
        rnn_units: int = 32,
        rnn_layer: str = "RNN",
        epochs: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        early_stopping_epochs: int | None = 100,
        random_seed: int = 42,
        feature_architecture: Expr | list[Expr] | dict | None = None,
        loss_on: str = "trajectory",
        first_step_weight: float | None = None,
        step_weights: list[float] | None = None,
        **kwargs,
    ):
        init_layer = kwargs.pop("init_layer", None)
        if init_layer not in (None, "out_model"):
            raise ValueError(
                "'init_layer' was removed in physXAI 1.0: the model warms itself up on the warmup sequence "
                "(as in AgentLib-MPC) instead of using a separate initialization model."
            )
        super().__init__(**kwargs)
        if loss_on not in ("trajectory", "outputs"):
            raise ValueError(f"Unknown loss_on '{loss_on}', use 'trajectory' or 'outputs'.")
        if first_step_weight is not None and not 0 < first_step_weight <= 1:
            raise ValueError("first_step_weight has to be in (0, 1].")

        self.rnn_units: int = rnn_units
        self.rnn_layer: str = rnn_layer
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.early_stopping_epochs: int | None = early_stopping_epochs
        self.random_seed: int = random_seed
        if isinstance(feature_architecture, dict):
            feature_architecture = expr_from_config(feature_architecture)
        if isinstance(feature_architecture, Expr):
            feature_architecture = [feature_architecture]
        self.feature_architecture: list[Expr] | None = feature_architecture
        self.loss_on: str = loss_on
        self.first_step_weight: float | None = first_step_weight
        self.step_weights: list[float] | None = list(step_weights) if step_weights is not None else None

        self.model_config = {"rnn_units": rnn_units, "rnn_layer": rnn_layer}

    @property
    def required_inputs(self) -> list[str] | None:
        """Inputs required by the feature architecture (None without one)."""
        if self.feature_architecture is None:
            return None
        names = list()
        for expression in self.feature_architecture:
            names.extend(v for v in expression.free_vars() if v not in names)
        return names

    def prepare(self):
        """Seeds keras before the model is generated, so weight initializations are reproducible."""
        keras.utils.set_random_seed(self.random_seed)

    def _spec(self, td: TrainingDataMultiStep) -> SequenceSpec:
        spec = SequenceSpec.from_training_data(td)
        if spec.feedback and not spec.absolute:
            missing = [f for f in td.features if f not in td.series.all_columns]
            raise ValueError(f"Fed back features need measured start values, the data holds no columns {missing}.")
        return spec

    def _trajectory_loss(self, td: TrainingDataMultiStep, warn: bool = False) -> bool:
        if warn and self.loss_on == "trajectory" and not td.absolute and any(td.difference):
            logger.warning(
                "The data holds no start values of the predicted features, the loss is computed on the outputs."
            )
        return self.loss_on == "trajectory"

    def generate_model(self, **kwargs):
        """
        Builds the step model and the sequence model used for training (which contains the step model).
        """
        td = kwargs["td"]
        step_model = build_step_model(self.model_config, td, feature_architecture=self.feature_architecture)
        return self._training_model(step_model, td)

    def _training_model(self, step_model, td: TrainingDataMultiStep):
        spec = self._spec(td)
        trajectory = self._trajectory_loss(td, warn=True)
        model = build_sequence_model(step_model, spec, td.warmup_width, td.label_width, trajectory=trajectory)
        # the loss is scaled per output, so outputs of different units contribute alike
        truth = sequence_inputs(td, "train", spec, trajectory)[1]
        self._scale = np.maximum(truth.reshape(-1, truth.shape[-1]).std(axis=0), 1e-6)
        if logger.isEnabledFor(logging.INFO):
            step_model.summary()
        return model

    def compile_model(self, model):
        """Compiles the sequence model with the step weighted squared error."""
        model.compile(
            keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=weighted_step_loss(self._weights(model), getattr(self, "_scale", None)),
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)],
        )

    def _weights(self, model) -> np.ndarray:
        horizon = int(model.outputs[0].shape[1])
        return step_weights(horizon, self.first_step_weight, self.step_weights)

    def fit_model(self, model, td: TrainingDataMultiStep):
        """
        Fits the sequence model; the training sequences are reshuffled in every epoch.

        Args:
            model (keras.Model): The sequence model (see `generate_model`).
            td (TrainingDataMultiStep): The training data.
        """
        spec = self._spec(td)
        trajectory = self._trajectory_loss(td)
        train = sequence_inputs(td, "train", spec, trajectory)
        val = sequence_inputs(td, "val", spec, trajectory)

        def dataset(data, shuffle: bool):
            ds = tf.data.Dataset.from_tensor_slices((tuple(data[0]) if isinstance(data[0], list) else data[0], data[1]))
            if shuffle:
                ds = ds.shuffle(len(data[1]), seed=self.random_seed, reshuffle_each_iteration=True)
            return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        callbacks = list()
        if self.early_stopping_epochs is not None and val is not None:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=self.early_stopping_epochs,
                restore_best_weights=True, verbose=keras_verbosity_int()))

        start_time = time.perf_counter()
        training_history = model.fit(
            dataset(train, shuffle=True),
            validation_data=dataset(val, shuffle=False) if val is not None else None,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=keras_verbosity(),
            shuffle=False,  # the dataset is reshuffled in every epoch
        )
        td.add_training_time(time.perf_counter() - start_time)
        td.add_training_record(training_history)

    def online_pipeline(self, td: TrainingDataMultiStep, load_path: str, save_path: str = None,
                        plot: bool = True, save_model: bool = True, show_plot: bool = False):
        """Loads a saved step model, trains it further on `td`, evaluates, plots and saves it."""
        self._check_training_data(td)
        model = self._training_model(self.load_model(load_path), td)
        self.compile_model(model)
        model = self.online_pipeline_internal(td, model)
        self._finish(model, td, save_path, plot, save_model, show_plot)
        return model

    def save_model(self, model, save_path: str):
        """
        Saves the step model `[sequence, *states] -> [prediction, *states]` as keras file.

        Args:
            model (keras.Model): The sequence model returned by the pipeline, or the step model.
            save_path (str): The path of the file ('.keras' is appended if missing).
        """
        find_step_model(model).save(output_path(save_path, '.keras'))

    def load_model(self, load_path: str):
        """
        Loads a saved step model.

        Args:
            load_path (str): The path of the keras file.

        Returns:
            keras.Model: The step model.
        """
        load_path = input_path(load_path)
        return find_step_model(keras.saving.load_model(load_path, compile=False))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'rnn_units': self.rnn_units,
            'rnn_layer': self.rnn_layer,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'early_stopping_epochs': self.early_stopping_epochs,
            'random_seed': self.random_seed,
            'feature_architecture': expr_to_config(self.feature_architecture)
                                    if self.feature_architecture is not None else None,
            'loss_on': self.loss_on,
            'first_step_weight': self.first_step_weight,
            'step_weights': self.step_weights,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'RNNModel':
        config = dict(config)
        config.pop('physXAI_version', None)
        config.pop('__class_name__', None)
        return cls(**config)
