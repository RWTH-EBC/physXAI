"""
Keras models of physXAI: single step networks (`ClassicalANNModel`, `CMNNModel`, `RBFModel`, `LinANNModel`,
`PINNModel`, base class `ANNModel`) and recurrent multi step models (`RNNModel`, `PCNNModel`).

All of them share the training settings (epochs, learning rate, batch size, early stopping, seed); single step networks
can additionally be trained on the multi step error (`recursive_training`) and with a physics residual (`physics`).
"""

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
    PCNNTerm,
    SequenceSpec,
    build_pcnn_step_model,
    build_sequence_model,
    build_step_model,
    find_step_model,
    pcnn_parameters,
    sequence_inputs,
)
from physXAI.models.ann.pinn.physics import PhysicsLoss, PhysicsResidual, fit_with_physics, physics_metrics
from physXAI.models.models import LinearRegressionModel, MultiStepModel, SingleStepModel, register_model
from physXAI.preprocessing.preprocessing import TARGET_PREFIX
from physXAI.preprocessing.training_data import TrainingData, TrainingDataGeneric, TrainingDataMultiStep
from physXAI.training.recursive import step_weights, weighted_step_loss
from physXAI.utils.seed import resolve_seed

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
                 early_stopping_epochs: int | None = 100, random_seed: int | None = None,
                 recursive_training: Union['RecursiveTraining', dict] | None = None,
                 physics: PhysicsResidual | dict | None = None, physics_fit: str = 'calibrate',
                 physics_learning_rate: float | None = None, **kwargs):
        """
        Initializes common hyperparameters for ANN training.

        Args:
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                         If None, early stopping is disabled.
            random_seed (int, optional): Seed of the weight initialization and the batch order, set when the model
                               is generated. Defaults to the global seed (`physXAI.set_seed`), else a new seed is
                               drawn. The used seed is stored in the config.
            recursive_training (RecursiveTraining, optional): Train on the multi step (recursive) error,
                               see `physXAI.training.RecursiveTraining`. None trains on the single step error.
            physics (PhysicsResidual, optional): A physics residual with trainable physical parameters, used in
                               the (single step) training, see `physXAI.models.ann.pinn.physics.PhysicsResidual`.
            physics_fit (str): How the parameters of the residual are trained: 'calibrate' (fitted to the data
                               first, then fixed while the network is trained), 'joint' or 'fixed'.
            physics_learning_rate (float, optional): Learning rate of the calibration (default 10 x learning_rate).
        """
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.early_stopping_epochs: int | None = early_stopping_epochs
        self.random_seed: int = resolve_seed(random_seed)
        from physXAI.training.recursive import RecursiveTraining
        if isinstance(recursive_training, dict):
            recursive_training = RecursiveTraining.from_config(recursive_training)
        self.recursive_training: RecursiveTraining | None = recursive_training
        if isinstance(physics, dict):
            physics = PhysicsResidual.from_config(physics)
        if physics_fit not in ('calibrate', 'joint', 'fixed'):
            raise ValueError(f"Unknown physics_fit '{physics_fit}', use 'calibrate', 'joint' or 'fixed'.")
        self.physics: PhysicsResidual | None = physics
        self.physics_fit: str = physics_fit
        self.physics_learning_rate: float | None = physics_learning_rate
        #: Trained values of the parameters of the physics residual (after the pipeline)
        self.physics_parameters: dict | None = None

        self.model_config = dict()

    @property
    def required_loss_columns(self) -> list[str] | None:
        """Data columns of the physics residual, added to the loss columns of the preprocessing."""
        return self.physics.columns if self.physics is not None else None

    def _check_training_data(self, td: TrainingDataGeneric):
        super()._check_training_data(td)
        if len(td.outputs) != 1:
            # the networks have one output neuron: several targets would be broadcast against it in the loss
            raise ValueError(
                f"{self.__class__.__name__} predicts one output, got {td.outputs}. Train one model per output "
                "(as AgentLib-MPC expects it), or use LinearRegressionModel or an RNNModel for several outputs."
            )
        if self.physics is not None:
            missing = [c for c in self.physics.columns if c not in td.loss_columns]
            if missing:
                raise ValueError(
                    f"The training data holds no columns {missing} of the physics residual. Pass the model to the "
                    "preprocessing, prep.pipeline(path, model=model), so it adds them as loss columns."
                )

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
        keras.utils.set_random_seed(self.random_seed % 2**32)

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

    def _fit_data(self, td: TrainingDataGeneric, split: str):
        """Inputs and targets the model is fitted on, None for a split without data."""
        X = getattr(td, f'X_{split}_single')
        return None if X is None else (X, getattr(td, f'y_{split}_single'))

    def _early_stopping_monitor(self) -> str:
        """Validation metric of the early stopping."""
        return 'val_loss'

    def fit_single_step(self, model, td: TrainingDataGeneric):
        """Fits the Keras model on the single step error (and the physics residual, if any)."""
        if self.physics is not None:
            start_time = time.perf_counter()
            result = fit_with_physics(model, td, self, self.physics, self.physics_fit, self.physics_learning_rate)
            td.add_training_time(time.perf_counter() - start_time)
            td.add_training_record(result.pop('record'))
            td.physics = result
            self.physics_parameters = result['parameters']
            # the network was trained inside a wrapper: build its own optimizer, so the saved model loads cleanly
            if getattr(model, 'optimizer', None) is not None and not model.optimizer.built:
                model.optimizer.build(model.trainable_variables)
            logger.info("Physics parameters: %s", self.physics_parameters)
            return
        train, val = self._fit_data(td, 'train'), self._fit_data(td, 'val')
        # Early stopping (only possible with validation data)
        callbacks = list()
        if self.early_stopping_epochs is not None and val is not None:
            es = keras.callbacks.EarlyStopping(monitor=self._early_stopping_monitor(), mode='min',
                                               patience=self.early_stopping_epochs,
                                               restore_best_weights=True, verbose=keras_verbosity_int())
            callbacks.append(es)

        # Fit model, track training time
        start_time = time.perf_counter()

        # Create tf.data.Dataset for better performance
        train_ds = tf.data.Dataset.from_tensor_slices(train)
        train_ds = (
            train_ds
            .cache()                                                                     # 1. Cache first so data is not reloaded, may need to be removed for large datasets
            .shuffle(buffer_size=min(10000, train[0].shape[0]), reshuffle_each_iteration=True)  # 2. Shuffle randomly every epoch
            .batch(self.batch_size)                                                      # 3. Group into batches
            .prefetch(buffer_size=tf.data.AUTOTUNE)                                      # 4. Prepare next batch in background
        )

        # Check for validation data
        if val is not None:
            val_ds = tf.data.Dataset.from_tensor_slices(val)
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
            'physics': self.physics.get_config() if self.physics is not None else None,
            'physics_fit': self.physics_fit,
            'physics_learning_rate': self.physics_learning_rate,
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
                 early_stopping_epochs: int | None = 100, random_seed: int | None = None, **kwargs):
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
              random_seed (int, optional): Seed of the weights and the batch order, defaults to the global seed.
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
    A hybrid model: a linear regression (fitted first, then frozen) plus an RBF network trained on its residuals.
    """

    def __init__(self, n_layers: int = 1, n_neurons: int | list[int] = 32, rescale_output: bool = True,
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int | None = 100, random_seed: int | None = None, **kwargs):
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
            random_seed (int, optional): Seed of the weights and the batch order, defaults to the global seed.
        """
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.rescale_output: bool = rescale_output

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output,
        })

    def generate_model(self, **kwargs):
        """
        Generates the hybrid Linear + ANN model.
        First, a Linear Regression model is trained. Then, an RBF network
        is constructed to model its residuals.
        """

        td = kwargs['td']

        # Fit the linear regression (only fitted: its evaluation would overwrite the results in td)
        lr_model = LinearRegressionModel().generate_model()
        lr_model.fit(td.X_train_single, td.y_train_single)

        # Construct residual model (the k-means seed is the seed of the model, also if it changed after __init__)
        config = {**self.model_config, "random_state": self.random_seed % 2**32}
        model = LinResidualANNConstruction(config, td, lr_model)
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
                 early_stopping_epochs: int | None = 100, random_seed: int | None = None, **kwargs):
        """
        Initializes the RBFModel.

        Args:
            n_layers (int): Number of RBF layers, only 1 is supported.
            n_neurons (int or list[int]): Number of RBF neurons (k-means centers).
            rescale_output (bool): Whether to rescale the model's output.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                           If None, early stopping is disabled.
            random_seed (int, optional): Seed of the weights and the batch order, defaults to the global seed.
        """
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.rescale_output: bool = rescale_output

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "rescale_output": self.rescale_output,
        })

    def generate_model(self, **kwargs):
        """
        Generates the Keras RBF model using RBFModelConstruction.
        """

        td = kwargs['td']
        # the k-means seed is the seed of the model, also if it changed after __init__ (repeat_training)
        model = RBFModelConstruction({**self.model_config, "random_state": self.random_seed % 2**32}, td)
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
                 early_stopping_epochs: int | None = 100, random_seed: int | None = None, **kwargs):
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
            random_seed (int, optional): Seed of the weights and the batch order, defaults to the global seed.
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
    Physics-informed neural network: a CMNN trained on the data and on physics targets.

    A physics target is a data column computed from a physical model of the output, e.g. a simplified
    heat pump model `columns={'pinn': expression}` in the preprocessing. The model has a single output
    (the data output) and is used and exported like any other model; the physics targets only enter the loss
    (see `physXAI.models.ann.pinn.physics`):

        L = mse(y_hat, y) / s + sum_i w_i * mse(y_hat, p_i) / s,   s = var(y_train)

    Physics targets are taken at the predicted time, like the output. For a model predicting `Change(T)`
    with a physics model of the step `T(t) -> T(t+1)`, the column of the physics prediction computed from the
    data at time t has to be shifted back by one step: `columns={'dT_rc': expression.lag(1)}`.

    Example:
        >>> prep = PreprocessingSingleStep(inputs=[...], output='P_el', columns={'pinn': physics})
        >>> m = PINNModel(physics_targets={'pinn': 0.5})
        >>> td = prep.pipeline(path, model=m)   # adds the physics targets as loss columns
        >>> m.pipeline(td)
    """

    plot_title = "Physics-Informed Neural Network"

    def __init__(self, physics_targets: dict[str, float], n_layers: int = 1, n_neurons: int | list[int] = 32,
                 activation_function: str | list[str] = 'softplus', rescale_output: bool = True,
                 monotonies: dict[str, int] = None, activation_split: list[float] = None,
                 normalize_terms: bool = True, early_stopping_on: str = 'data',
                 batch_size: int = 32, epochs: int = 1000, learning_rate: float = 0.001,
                 early_stopping_epochs: int | None = 100, random_seed: int | None = None, **kwargs):
        """
        Initializes the PINNModel.

        Args:
            physics_targets (dict[str, float]): {column: weight} of the physics targets. The weight is the
                                                relative importance of the term compared to the data.
            n_layers (int): Number of hidden layers.
            n_neurons (int or list[int]): Number of neurons per layer.
            activation_function (str or list[str]): Activation function(s).
            rescale_output (bool): Whether to rescale output.
            monotonies (dict[str, int]): Dictionary mapping feature names to monotonicity type
                                         (-1 for decreasing, 0 for no constraint, 1 for increasing).
            activation_split (list[float]): Proportions for splitting neurons into convex,
                                            concave, and saturated activation paths. E.g., [0.5, 0.25, 0.25].
            normalize_terms (bool): Divide all terms of the loss by the variance of the training targets.
            early_stopping_on (str): 'data' (validation data loss, default) or 'total' (validation loss
                                     including the weighted physics terms).
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                           If None, early stopping is disabled.
            random_seed (int, optional): Seed of the weights and the batch order, defaults to the global seed.
        """
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        if not physics_targets:
            raise ValueError("PINNModel needs physics targets, e.g. physics_targets={'pinn': 1.0}.")
        if self.physics is not None:
            raise ValueError("PINNModel uses physics targets; for a physics residual use another model, e.g. "
                             "CMNNModel(physics=...).")
        if early_stopping_on not in ('data', 'total'):
            raise ValueError("early_stopping_on has to be 'data' or 'total'.")
        self.physics_targets: dict[str, float] = {str(k): float(w) for k, w in physics_targets.items()}
        self.n_layers: int = n_layers
        self.n_neurons: int | list[int] = n_neurons
        self.activation_function: str | list[str] = activation_function
        self.rescale_output: bool = rescale_output
        self.monotonies: dict[str, int] = monotonies
        self.activation_split: list[float] = activation_split
        self.normalize_terms: bool = normalize_terms
        self.early_stopping_on: str = early_stopping_on
        self._scale: float = 1.0

        self.model_config.update({
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation_function": self.activation_function,
            "rescale_output": self.rescale_output,
            "monotonicities": self.monotonies,
            "activation_split": activation_split,
        })

    @property
    def required_loss_columns(self) -> list[str]:
        """The physics targets at the predicted time, added to the loss columns of the preprocessing."""
        return [TARGET_PREFIX + name for name in self.physics_targets]

    def _check_training_data(self, td: TrainingDataGeneric):
        super()._check_training_data(td)
        missing = [c for c in self.required_loss_columns if c not in td.loss_columns]
        if missing:
            raise ValueError(
                f"The training data holds no physics targets {[c[len(TARGET_PREFIX):] for c in missing]}. "
                "Pass the model to the preprocessing, prep.pipeline(path, model=model), so it adds them."
            )
        if len(td.outputs) != 1:
            raise ValueError(f"PINNModel predicts one output, got {td.outputs}. Physics targets are no outputs.")
        variance = float(np.var(td.y_train_single[:, 0])) if self.normalize_terms else 1.0
        self._scale = max(variance, 1e-12)

    def generate_model(self, **kwargs):
        """
        Generates the Keras model (a CMNN) with one output.
        """
        td = kwargs['td']
        return CMNNModelConstruction(self.model_config, td)

    def compile_model(self, model):
        """
        Compiles the model with the physics loss and the metrics of every term (`data_loss`, `<target>_loss`).
        """
        model.compile(keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=PhysicsLoss(list(self.physics_targets.values()), self._scale),
                      metrics=physics_metrics(list(self.physics_targets), self._scale))

    def _fit_data(self, td: TrainingDataGeneric, split: str):
        data = super()._fit_data(td, split)
        if data is None:
            return None
        targets = td.values(split, self.required_loss_columns)
        return data[0], np.concatenate([data[1], targets], axis=1)

    def _early_stopping_monitor(self) -> str:
        return 'val_data_loss' if self.early_stopping_on == 'data' else 'val_loss'

    def evaluate(self, model, td: TrainingDataGeneric):
        """
        Evaluates the model like any single step model and adds the disagreement with the physics targets
        (`MetricsPINN`).
        """
        SingleStepModel.evaluate(model, td)
        physics = {
            name: {split: td.values(split, [column]) for split in ('train', 'val', 'test') if td.has_split(split)}
            for name, column in zip(self.physics_targets, self.required_loss_columns, strict=True)
        }
        td.add_metrics(MetricsPINN(td, physics, self._scale))

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
            "physics_targets": self.physics_targets,
            "normalize_terms": self.normalize_terms,
            "early_stopping_on": self.early_stopping_on,
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
        random_seed: Seed of the weight initialization and the batch order, set when the model is generated.
                     Defaults to the global seed (`physXAI.set_seed`), else a new seed is drawn (stored in the config).
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
        random_seed: int | None = None,
        feature_architecture: Expr | list[Expr] | dict | None = None,
        loss_on: str = "trajectory",
        first_step_weight: float | None = None,
        step_weights: list[float] | None = None,
        **kwargs,
    ):
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
        self.random_seed: int = resolve_seed(random_seed)
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
        keras.utils.set_random_seed(self.random_seed % 2**32)

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
        self.prepare()
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

@register_model
class PCNNModel(RNNModel):
    """
    Physically consistent neural network (PCNN, Di Natale et al. 2022, Applied Energy 325, 119806) for the
    multi step prediction of a temperature.

    The temperature T is the sum of two states: an energy accumulator E, a linear physics-inspired module
    driven by the heating and cooling inputs and the heat losses to the outside (or neighbouring zones), and the
    unforced dynamics D, a recurrent network f on D and the remaining inputs (e.g. solar radiation):

        E(k+1) = E(k) + sum a * u_heating(k) - sum d * u_cooling(k) + sum b * (T_x(k) - T(k))
        D(k+1) = D(k) + f(D(k), x(k)),   T = D + E

    Since D does not depend on the controls and T_x, the prediction is physically consistent over the whole
    horizon for any f: T increases with heating, decreases with cooling and increases with T_x, if the
    parameters a, b, d are positive (enforced) and 1 - sum b > 0 per time step (checked after the training).

    The data are `PreprocessingMultiStep` sequences with the output `px.Change('T')` and T as input; T is
    fed back during the prediction (closed loop), the model warms up on the measured past, and the parameters
    are trained with the network on the error of the trajectory over the horizon. The saved step model is
    exported to AgentLib-MPC like an RNN.

    The parameters start at initial rates per hour, scaled to the time step; the network learns a factor on them.
    A rate is either a physical rule of thumb (a heating rate in K/h at u = 1, e.g. "+1 K in 2 h at full power":
    0.5, a loss rate in 1/h, e.g. "-1.5 K in 6 h when it is 25 K colder outside": 1.5 / 25 / 6 = 0.01) or None:
    estimated from the training data by nonnegative least squares of the single step energy balance
    `T(k+1) - T(k) = sum a * u_heating - sum d * u_cooling + sum b * (T_x - T) + c` on the steps of the training
    split (the offset c belongs to the unforced dynamics D). The initial rates are reported in
    `td.physics['initial_rates']`.

    Args:
        heating: {input: initial heating rate in K/h per unit of the input, or None (from the data)}.
        cooling: {input: initial cooling rate in K/h per unit of the input, or None} (the input is a positive
                 cooling power).
        losses: {temperature input: initial loss rate in 1/h, or None}.
        rnn_units: Units of the recurrent layer of f.
        rnn_layer: 'LSTM' (default), 'GRU' or 'RNN'.
        **kwargs: Training settings of `RNNModel` (epochs, learning_rate, batch_size, early_stopping_epochs,
                  random_seed, loss_on, first_step_weight, step_weights).

    Example:
        >>> prep = PreprocessingMultiStep(inputs=['u_hp', 'T_amb', 'solar', 'T'], output=px.Change('T'),
        ...                               warmup_width='3h', label_width='12h')
        >>> m = PCNNModel(heating={'u_hp': 0.5}, losses={'T_amb': 0.01})   # rules of thumb
        >>> m = PCNNModel(heating={'u_hp': None}, losses={'T_amb': None})  # estimated from the training data
    """

    plot_title = "Physically Consistent Neural Network"

    def __init__(self, heating: dict[str, float] | None = None, cooling: dict[str, float] | None = None,
                 losses: dict[str, float] | None = None, rnn_units: int = 32, rnn_layer: str = "LSTM", **kwargs):
        if kwargs.get("feature_architecture") is not None:
            raise ValueError("PCNNModel has no feature_architecture, the inputs of f are all non-physics inputs.")
        super().__init__(rnn_units=rnn_units, rnn_layer=rnn_layer, **kwargs)
        def rates_of(values):
            return {str(k): (None if v is None else float(v)) for k, v in (values or {}).items()}

        self.heating = rates_of(heating)
        self.cooling = rates_of(cooling)
        self.losses = rates_of(losses)
        rates = {**self.heating, **self.cooling, **self.losses}
        if not rates:
            raise ValueError("PCNNModel needs at least one term: heating, cooling or losses.")
        if any(rate is not None and rate <= 0 for rate in rates.values()):
            raise ValueError(f"The initial rates have to be positive (or None: from the data), got {rates}.")
        self._estimated: tuple[int, dict[str, float]] | None = None  # (id of the series, rates per hour)
        names = [*self.heating, *self.cooling, *self.losses]
        if len(set(names)) != len(names):
            raise ValueError(f"Every input can only be part of one term, got {names}.")

    def _terms(self, td: TrainingDataMultiStep) -> list[PCNNTerm]:
        """The terms of the energy accumulator with their initial values per time step."""
        if len(td.outputs) != 1 or not td.difference[0]:
            raise ValueError(f"PCNNModel predicts the change of one feature (px.Change('T')), got {td.outputs}.")
        feature = td.features[0]
        if feature not in td.inputs:
            raise ValueError(f"The predicted feature '{feature}' has to be an input of the PCNN (closed loop).")
        if not td.time_step:
            raise ValueError("PCNNModel needs the time step of the data to scale its rates.")
        hours = td.time_step / 3600.0
        kinds = (("heating", self.heating), ("cooling", self.cooling), ("loss", self.losses))
        for kind, rates in kinds:
            for name in rates:
                if name not in td.inputs:
                    raise ValueError(f"The {kind} input '{name}' is not an input of the data {td.inputs}.")
                if name == feature:
                    raise ValueError(f"The predicted feature '{feature}' cannot be a {kind} input.")
        initial = self.initial_rates(td)
        return [PCNNTerm(kind, td.inputs.index(name), initial[name] * hours) for kind, rates in kinds for name in rates]

    def initial_rates(self, td: TrainingDataMultiStep) -> dict[str, float]:
        """
        Initial rates per hour of all terms: the given ones, and the ones given as None estimated from the steps of
        the training split by nonnegative least squares (see the class docstring).
        """
        given = {**self.heating, **self.cooling, **self.losses}
        fixed = {k: v for k, v in given.items() if v is not None}
        if len(fixed) == len(given):
            return fixed
        if self._estimated is not None and self._estimated[0] == id(td.series):
            return {**self._estimated[1], **fixed}
        from physXAI.models.modular.shape_constraints import constrained_least_squares
        from physXAI.preprocessing.windows import gather_windows

        series = td.series
        starts = series.window_starts("train", 2, 1)
        pairs = gather_windows(series.data[:, series._input_index], starts, 2).astype(np.float64)
        pairs = pairs[np.all(np.isfinite(pairs), axis=(1, 2))]
        if len(pairs) < 10:
            raise ValueError("Too few steps in the training data to estimate the initial rates of the PCNN.")
        t = td.inputs.index(td.features[0])
        now = pairs[:, 0, :]
        change = pairs[:, 1, t] - pairs[:, 0, t]
        columns, names = list(), list()
        for sign, rates in ((1.0, self.heating), (-1.0, self.cooling)):
            for name in rates:
                columns.append(sign * now[:, td.inputs.index(name)])
                names.append(name)
        for name in self.losses:
            columns.append(now[:, td.inputs.index(name)] - now[:, t])
            names.append(name)
        features = np.column_stack(columns)
        weights, _ = constrained_least_squares(features, change, bias=True, weight_signs=[1] * len(names))
        hours = td.time_step / 3600.0
        estimated = dict()
        for k, name in enumerate(names):
            # a rate of 0 would switch the term off for good (the network only learns a factor): start at a weak one
            floor = 1e-2 * float(np.std(change)) / max(float(np.std(features[:, k])), 1e-12)
            if weights[k] < floor:
                logger.warning("PCNN: the training data show no positive effect of '%s' (estimated %.3g per step), "
                               "it starts at the weak rate %.3g per step.", name, weights[k], floor)
            estimated[name] = max(float(weights[k]), floor) / hours
        self._estimated = (id(td.series), estimated)
        logger.info("PCNN initial rates estimated from the training data (per hour): %s",
                    {k: v for k, v in estimated.items() if given[k] is None})
        return {**estimated, **fixed}

    def _check_training_data(self, td: TrainingDataGeneric):
        super()._check_training_data(td)
        self._terms(td)

    def generate_model(self, **kwargs):
        """Builds the PCNN step model and the sequence model used for training."""
        td = kwargs["td"]
        step_model = build_pcnn_step_model(self.model_config, td, self._terms(td))
        return self._training_model(step_model, td)

    def evaluate(self, model, td: TrainingDataMultiStep):
        """Evaluates the model like an RNN and stores the trained physical parameters in `td.physics`."""
        MultiStepModel.evaluate(model, td)
        td.physics = {"parameters": self.physics_parameters(model, td), "initial_rates": self.initial_rates(td)}

    def physics_parameters(self, model, td: TrainingDataMultiStep) -> dict:
        """
        Trained parameters of the energy accumulator per hour (heating and cooling rates in K/h per unit of
        the input, loss rates in 1/h) and the check of the physical consistency 1 - sum(loss rates per step) > 0.
        """
        terms = self._terms(td)
        per_step = pcnn_parameters(find_step_model(model), terms)
        hours = td.time_step / 3600.0
        kinds = {"heating": "heating", "cooling": "cooling", "loss": "losses"}
        parameters = {"heating": {}, "cooling": {}, "losses": {}}
        for term, value in zip(terms, per_step, strict=True):
            parameters[kinds[term.kind]][td.inputs[term.input]] = value / hours
        margin = 1.0 - sum(v for term, v in zip(terms, per_step, strict=True) if term.kind == "loss")
        parameters["consistency_margin"] = margin
        if margin <= 0:
            logger.warning("PCNN: the loss rates per time step sum up to %.3g >= 1, the prediction is not "
                           "physically consistent anymore.", 1.0 - margin)
        logger.info("PCNN parameters (per hour): %s", parameters)
        return parameters

    def get_config(self) -> dict:
        config = super().get_config()
        config.pop("feature_architecture", None)
        config.update({"heating": self.heating, "cooling": self.cooling, "losses": self.losses})
        return config

