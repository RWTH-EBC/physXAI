import time
from abc import ABC, abstractmethod
from typing import Type
import joblib
import numpy as np
from joblib import dump
from sklearn.linear_model import LinearRegression
from physXAI.utils.logging import get_full_path, create_full_path, Logger
from physXAI.preprocessing.training_data import TrainingData, TrainingDataMultiStep, TrainingDataGeneric
from physXAI.evaluation.metrics import Metrics, MetricsMultiStep
from physXAI.plotting.plotting import (plot_prediction_correlation, plot_metrics_table, subplots,
                                              plot_predictions, plot_multi_rmse)


MODEL_CLASS_REGISTRY: dict[str, Type['AbstractModel']] = dict()


class AbstractModel(ABC):

    @abstractmethod
    def generate_model(self, **kwargs):
        """
        Abstract method to be implemented by subclasses.
        Should generate and return an instance of the specific model.
        `kwargs` can be used to pass necessary parameters, e.g., `td` (TrainingData or TrainingDataMultiStep).
        """
        pass

    @abstractmethod
    def compile_model(self, model):
        """
        Abstract method for model compilation.
        Relevant for models like Keras neural networks. For scikit-learn models,
        this might be a pass-through or not applicable.
        """
        pass

    @abstractmethod
    def fit_model(self, model, td: TrainingDataGeneric):
        """
        Abstract method to fit the model to the training data.

        Args:
            model: The model instance to be trained.
            td (TrainingDataGeneric): The TrainingData object.
        """
        pass

    @staticmethod
    @abstractmethod
    def evaluate(model, td: TrainingDataGeneric, **kwargs):
        pass

    @abstractmethod
    def plot(self, td: TrainingDataGeneric):
        """
         Abstract method for generating and displaying plots related to model performance.

         Args:
             td (TrainingDataGeneric): The TrainingData object containing true values and predictions.
         """
        pass

    @abstractmethod
    def save_model(self, model, save_path: str):
        """
        Abstract method for saving the trained model.

        Args:
            model: The trained model instance to save.
            save_path (str): The path where the model should be saved.
        """
        pass

    @abstractmethod
    def load_model(self, load_path: str):
        """
        Abstract method for loading a pre-trained model.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            The loaded model instance.
        """
        pass

    def pipeline(self, td: TrainingDataGeneric, save_path: str = None, plot: bool = True, save_model: bool = True,
                 **kwargs):
        """
          Defines a standard pipeline for single-step models:
          1. Generate model
          2. Compile model (if applicable)
          3. Fit model
          4. Evaluate model
          5. Plot results
          6. Save model (if save_path is provided)

          Args:
              td (TrainingDataGeneric): The training data.
              save_path (str, optional): Path to save the trained model. Defaults to None (Saving path from Logger).
              plot (bool, optional): Whether to plot the results. Defaults to True.
              save_model (bool, optional): Whether to save the trained model. Defaults to True.

          Returns:
              The trained model instance.
          """

        model = self.generate_model(td=td)
        self.compile_model(model)
        self.fit_model(model, td)

        self.evaluate(model, td, **kwargs)
        if plot:
            self.plot(td)
        if save_model:
            self.save_model(model, save_path=save_path)

        return model

    def online_pipeline(self, td: TrainingDataGeneric, load_path: str, save_path: str = None,
                        plot: bool = True, save_model: bool = True):
        """
        Implements an "online" training pipeline: loads a pre-existing model,
        further trains it on new data, evaluates, plots, and saves it back.

        Args:
            td (TrainingDataGeneric): New training data.
            load_path (str): Path to the pre-existing model.
            save_path (str, optional): Path to save the trained model. Defaults to None (Saving path from Logger).
            plot (bool, optional): Whether to plot the results. Defaults to True.
            save_model (bool, optional): Whether to save the trained model. Defaults to True.

        Returns:
            The updated and saved model.
        """

        model = self.load_model(load_path)
        self.compile_model(model)

        model = self.online_pipeline_internal(td, model)

        if plot:
            self.plot(td)
        if save_model:
            self.save_model(model, save_path=save_path)

        return model

    def online_pipeline_internal(self, td: TrainingDataGeneric, model):
        """
        Implements an "online" training pipeline: trains a pre-existing model on new data

        Args:
            td (TrainingDataGeneric): New training data.
            model: The model to train.

        Returns:
            The updated and saved model.
        """
        self.fit_model(model, td)
        self.evaluate(model, td)

        return model

    def get_config(self) -> dict:
        return {
            '__class_name__': self.__class__.__name__,
        }

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> 'AbstractModel':
        pass

    @staticmethod
    def model_from_config(item_conf: dict) -> 'AbstractModel':
        """
        Factory function to create a model object from its configuration dictionary.

        Args:
            item_conf (dict): The configuration dictionary for a model.
                              Must contain 'class_name' and other necessary parameters.

        Returns:
            AbstractModel: An instance of the appropriate model subclass.

        Raises:
            KeyError: If 'class_name' is not in `item_conf` or if the class_name is not in `MODEL_CLASS_REGISTRY`.
        """
        class_name = item_conf['__class_name__']
        model_class = MODEL_CLASS_REGISTRY[class_name]
        m = model_class.from_config(item_conf)
        return m


def register_model(cls):  # pragma: no cover
    """
    A class decorator that registers the decorated class in the MODEL_CLASS_REGISTRY.
    The class is registered using its __name__.
    """
    if cls.__name__ in MODEL_CLASS_REGISTRY:
        print(f"Warning: Class '{cls.__name__}' is already registered. Overwriting.")
    MODEL_CLASS_REGISTRY[cls.__name__] = cls
    return cls  # Decorators must return the class (or a replacement)


class SingleStepModel(AbstractModel, ABC):
    """
    Abstract Base Class for single-step prediction models.
    Defines a common interface and a pipeline for training, evaluating,
    plotting, and saving models that predict a single output based on input features.
    """

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def evaluate(model, td: TrainingDataGeneric, **kwargs):
        """
        Evaluates the trained model on training, validation (if available), and test sets.
        Predictions are stored in the TrainingData object, and metrics are calculated and stored.

        Args:
            model: The trained model instance.
            td (TrainingDataGeneric): The TrainingData object containing datasets and for storing results.
        """

        y_pred_train = model.predict(td.X_train_single)
        y_pred_test = model.predict(td.X_test_single)
        if td.X_val_single is not None:
            y_pred_val = model.predict(td.X_val_single)
        else:
            y_pred_val = None
        td.add_predictions(y_pred_train, y_pred_val, y_pred_test)

        metrics = Metrics(td)
        if isinstance(td, TrainingData):
            td.add_metrics(metrics)
        elif isinstance(td, TrainingDataMultiStep):
            td.add_single_step_metrics(metrics)
            SingleStepModel.evaluate_multi(model, td)
        else:
            raise NotImplementedError

    @staticmethod
    def evaluate_multi(model, td: TrainingDataMultiStep):
        if td.init_columns[0] != td.output[0]:
            delta_prediction = True
        else:
            delta_prediction = False
        y_pred_train, y_train = SingleStepModel._evaluate_multi_inner_loop(model, td.X_train_features, td.y_train,
                                                                           td.columns, td.init_columns[0],
                                                                           delta_prediction)
        y_pred_test, y_test = SingleStepModel._evaluate_multi_inner_loop(model, td.X_test_features, td.y_test,
                                                                         td.columns, td.init_columns[0],
                                                                         delta_prediction)
        if td.X_val_features is not None:
            y_pred_val, y_val = SingleStepModel._evaluate_multi_inner_loop(model, td.X_val_features, td.y_val,
                                                                           td.columns, td.init_columns[0],
                                                                           delta_prediction)
        else:
            y_pred_val, y_val = None, None
        td.add_predictions(y_pred_train, y_pred_val, y_pred_test)
        td.y_train = y_train
        td.y_val = y_val
        td.y_test = y_test

        metrics = MetricsMultiStep(td)
        td.add_metrics(metrics)

    @staticmethod
    def _evaluate_multi_inner_loop(model, X: np.ndarray, y: np.ndarray, X_columns: list[str],
                                   recursive_output_column: str, delta_prediction: bool = True) \
            -> (np.ndarray, np.ndarray):
        true_vals = np.ndarray(shape=(y.shape[0], y.shape[1], 1), dtype=np.float64)
        preds = np.ndarray(shape=(X.shape[0], X.shape[1], 1), dtype=np.float64)

        assert recursive_output_column in X_columns, (f'Error: Cannot find recursive_output_column '
                                                      f'"{recursive_output_column}" in X_columns')
        if isinstance(model, LinearRegression):
            for b in range(X.shape[0]):
                index = X_columns.index(recursive_output_column)
                current_val = X[b, 0, index]
                current_true_val = current_val
                for t in range(X.shape[1]):
                    pred = float(model.predict(X[b, t, :].reshape(1, -1)))
                    if delta_prediction:
                        current_val += pred
                        current_true_val += y[b, t, 0]
                        pred = current_val
                    else:
                        current_true_val = y[b, t, 0]
                    preds[b, t, 0] = pred
                    true_vals[b, t, 0] = current_true_val

                    for l in range(0, 10):
                        if l == 0:
                            col = recursive_output_column
                        else:
                            col = recursive_output_column + f'_lag{l}'
                        if col in X_columns:
                            index = X_columns.index(col)
                            if t + 1 + l < X.shape[1]:
                                X[b, t + 1 + l, index] = pred
        else:
            index = X_columns.index(recursive_output_column)
            current_val = X[:, 0, index].reshape(-1, 1)
            current_true_val = current_val.copy()
            for t in range(X.shape[1]):
                pred = model.predict(X[:, t, :], verbose=0)
                if delta_prediction:
                    current_val += pred
                    current_true_val += y[:, t, 0].reshape(-1, 1)
                    pred = current_val
                else:
                    current_true_val = y[:, t, 0].reshape(-1, 1)
                preds[:, t, 0] = pred.reshape(-1)
                true_vals[:, t, 0] = current_true_val.reshape(-1)

                for l in range(0, 10):
                    if l == 0:
                        col = recursive_output_column
                    else:
                        col = recursive_output_column + f'_lag{l}'
                    if col in X_columns:
                        index = X_columns.index(col)
                        if t + 1 + l < X.shape[1]:
                            X[:, t + 1 + l, index] = pred.reshape(-1)
        return preds, true_vals

    @classmethod
    def from_config(cls, config: dict) -> 'SingleStepModel':
        return cls(**config)


@register_model
class LinearRegressionModel(SingleStepModel):
    """
    A concrete implementation of SingleStepModel for scikit-learn's Linear Regression.
    """

    def generate_model(self, **kwargs):
        """
        Generates an instance of scikit-learn's LinearRegression model.
        """
        return LinearRegression()

    def fit_model(self, model, td: TrainingDataGeneric):
        """
        Fits the LinearRegression model using the training data from `td`.
        Also records the training time.

        Args:
            model (LinearRegression): The scikit-learn LinearRegression model instance.
            td (TrainingDataGeneric): The TrainingData object.
        """

        start_time = time.perf_counter()
        model.fit(td.X_train_single, td.y_train_single)
        stop_time = time.perf_counter()
        td.add_training_time(stop_time - start_time)

    def compile_model(self, model):
        """
        No compilation step is needed for scikit-learn models.
        """
        pass

    def plot(self, td: TrainingDataGeneric):
        """
        Generates and displays various plots related to model performance.

        Args:
            td (TrainingDataGeneric): The TrainingData object
        """

        fig1 = plot_prediction_correlation(td)
        fig2 = plot_predictions(td)
        fig3 = plot_metrics_table(td)

        if isinstance(td, TrainingData):
            subplots(
                "Linear Regression",
                {"title": "Prediction Correlation", "type": "scatter", "figure": fig1},
                {"title": "Predictions Sorted", "type": "scatter", "figure": fig2},
                {"title": "Performance Metrics", "type": "table", "figure": fig3}
            )
        elif isinstance(td, TrainingDataMultiStep):
            fig4 = plot_multi_rmse(td)
            subplots(
                "Linear Regression",
                # {"title": "Prediction Correlation", "type": "scatter", "figure": fig1},
                {"title": "Predictions Sorted", "type": "scatter", "figure": fig2},
                {"title": "Prediction Step RMSE", "type": "scatter", "figure": fig4},
                {"title": "Performance Metrics", "type": "table", "figure": fig3}
            )
        else:
            raise NotImplementedError

    def save_model(self, model, save_path: str):
        """
        Saves the trained LinearRegression model using joblib.

        Args:
            model (LinearRegression): The trained scikit-learn model.
            save_path (str): The path to save the model.
        """

        if save_path is None:
            save_path = Logger.get_model_savepath()

        if not save_path.endswith('.joblib'):
            save_path += '.joblib'

        save_path = create_full_path(save_path)
        dump(model, save_path)

    def load_model(self, load_path: str):
        """
        Loads a scikit-learn LinearRegression model from a file using joblib.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            LinearRegression: The loaded scikit-learn model.
        """

        load_path = get_full_path(load_path)
        model = joblib.load(load_path)
        return model


class MultiStepModel(AbstractModel, ABC):
    """
    Abstract Base Class for multi-step prediction models.
    Defines a common interface and pipeline for models that forecast multiple steps ahead.
    This class is similar to SingleStepModel but tailored for multi-step data and metrics.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit_model(self, model, td: TrainingDataMultiStep):
        """
        Abstract method to fit the model to the training data.

        Args:
            model: The model instance to be trained.
            td (TrainingDataMultiStep): The TrainingData object.
        """
        pass

    @staticmethod
    def evaluate(model, td: TrainingDataMultiStep, clipped_warmup_length: int = 0):
        """
        Evaluates the trained model on training, validation (if available), and test sets.
        Predictions are stored in the TrainingDataMultiStep object, and metrics are calculated and stored.

        Args:
            model: The trained model instance.
            td (TrainingDataMultistep): The TrainingDataMultiStep object containing datasets and for storing results.
            clipped_warmup_length (int): The clipped warmup length of the time series is not included in the evaluation sequence. Default is 0.
        """

        y_pred_train = model.predict(td.X_train)
        if td.X_val is not None:
            y_pred_val = model.predict(td.X_val)
        else:
            y_pred_val = None
        y_pred_test = model.predict(td.X_test)

        if clipped_warmup_length > 0:
            y_pred_train = y_pred_train[:, clipped_warmup_length:, :]
            if y_pred_val is not None:
                y_pred_val = y_pred_val[:, clipped_warmup_length:, :]
            y_pred_test = y_pred_test[:, clipped_warmup_length:, :]

            td.y_train = td.y_train[:, clipped_warmup_length:, :]
            if td.X_val is not None:
                td.y_val = td.y_val[:, clipped_warmup_length:, :]
            td.y_test = td.y_test[:, clipped_warmup_length:, :]

        td.add_predictions(y_pred_train, y_pred_val, y_pred_test)

        metrics = MetricsMultiStep(td)
        td.add_metrics(metrics)

    @abstractmethod
    def plot(self, td: TrainingDataMultiStep):
        """
         Abstract method for generating and displaying plots related to model performance.

         Args:
             td (TrainingDataMultiStep): The TrainingDataMultiStep object containing true values and predictions.
         """
        pass

    @classmethod
    def from_config(cls, config: dict) -> 'MultiStepModel':
        return cls(**config)
