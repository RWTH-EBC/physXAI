import logging
import os
import time
from abc import ABC, abstractmethod

import joblib
from joblib import dump
from sklearn.linear_model import LinearRegression

from physXAI.evaluation.metrics import Metrics
from physXAI.io.run import input_path, keras_verbosity, output_path
from physXAI.plotting.plotting import model_report, save_figure
from physXAI.preprocessing.training_data import TrainingData, TrainingDataGeneric, TrainingDataMultiStep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


logger = logging.getLogger("physXAI")

MODEL_CLASS_REGISTRY: dict[str, type['AbstractModel']] = dict()


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
    def evaluate(model, td: TrainingDataGeneric):
        pass

    #: Title of the plots of the model
    plot_title: str = "Model"

    def plot(self, td: TrainingDataGeneric, show: bool = False):
        """
        Plots the predictions, the errors (over the horizon), the training history and the metrics.

        Args:
            td (TrainingDataGeneric): The evaluated training data.
            show (bool): Whether to open the figure.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        return model_report(td, self.plot_title, show=show)

    @staticmethod
    def save_plot(figure, save_path: str | None = None):
        """
        Saves the figure of `plot` as html file next to the model: `<save_path>.html`.
        Nothing is saved without a path (use `Run.save` to save a model with its report).
        """
        if save_path is None:
            return None
        save_path = str(save_path)
        stem = os.path.splitext(save_path)[0] if save_path.endswith((".keras", ".joblib")) else save_path
        return save_figure(figure, output_path(stem + ".html"))

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

    def prepare(self):  # noqa: B027 - optional hook
        """Hook called before the model is generated (e.g. to set random seeds)."""

    def _check_training_data(self, td: TrainingDataGeneric):  # noqa: B027 - optional hook
        """Hook checking that the training data fits the model, called before the model is generated."""

    #: The trained model of the last `pipeline` or `online_pipeline` (used by `Run.save`)
    trained_model = None

    def pipeline(self, td: TrainingDataGeneric, save_path: str = None, plot: bool = True, save_model: bool = True,
                 show_plot: bool = False):
        """
        Trains and evaluates the model:

        1. Generate the model
        2. Compile the model (if applicable)
        3. Fit the model
        4. Evaluate the model, the metrics are stored in `td`
        5. Save the model and its report, if a `save_path` is given, and show the report if requested

        To save the model together with its configs and metrics, use `Run.save(name, preprocessing, model, td)`
        after the pipeline instead of `save_path`.

        Args:
            td (TrainingDataGeneric): The training data.
            save_path (str, optional): Path of the model file. Without a path nothing is saved.
            plot (bool, optional): Whether to save the report as html next to the model. Defaults to True.
            save_model (bool, optional): Whether to save the trained model (if `save_path` is given). Defaults to True.
            show_plot (bool, optional): Whether to open the report. Defaults to False.

        Returns:
            The trained model instance (also available as `self.trained_model`).
        """

        self._check_training_data(td)
        self.prepare()
        model = self.generate_model(td=td)
        self.compile_model(model)
        self.fit_model(model, td)

        self.evaluate(model, td)
        self.trained_model = model
        self._finish(model, td, save_path, plot, save_model, show_plot)
        return model

    def online_pipeline(self, td: TrainingDataGeneric, load_path: str, save_path: str = None,
                        plot: bool = True, save_model: bool = True, show_plot: bool = False):
        """
        Loads a saved model, trains it further on new data, evaluates and (optionally) saves it.

        Args:
            td (TrainingDataGeneric): New training data.
            load_path (str): Path to the saved model.
            save_path (str, optional): Path of the model file. Without a path nothing is saved.
            plot (bool, optional): Whether to save the report as html next to the model. Defaults to True.
            save_model (bool, optional): Whether to save the trained model (if `save_path` is given). Defaults to True.
            show_plot (bool, optional): Whether to open the report. Defaults to False.

        Returns:
            The trained model.
        """

        self._check_training_data(td)
        model = self.load_model(load_path)
        self.compile_model(model)

        model = self.online_pipeline_internal(td, model)
        self._finish(model, td, save_path, plot, save_model, show_plot)
        return model

    def online_pipeline_internal(self, td: TrainingDataGeneric, model):
        """
        Trains a loaded model on new data and evaluates it.

        Args:
            td (TrainingDataGeneric): New training data.
            model: The model to train.

        Returns:
            The trained model.
        """
        self.fit_model(model, td)
        self.evaluate(model, td)
        self.trained_model = model
        return model

    def _finish(self, model, td: TrainingDataGeneric, save_path: str | None, plot: bool, save_model: bool,
                show_plot: bool):
        """Saves the model and its report next to it (with a `save_path`) and shows the report if requested."""
        if show_plot or (plot and save_path is not None):
            self.save_plot(self.plot(td, show=show_plot), save_path if plot else None)
        if save_model and save_path is not None:
            self.save_model(model, save_path=save_path)

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
        logger.warning("Class '%s' is already registered. Overwriting.", cls.__name__)
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
    def evaluate(model, td: TrainingData):
        """
        Evaluates the trained model on training, validation (if available), and test sets.
        Predictions are stored in the TrainingData object, and metrics are calculated and stored.

        Args:
            model: The trained model instance.
            td (TrainingDataGeneric): The TrainingData object containing datasets and for storing results.
        """
        kwargs = {}
        if isinstance(model, keras.Model):
            kwargs['verbose'] = keras_verbosity()

        y_pred_train = model.predict(td.X_train_single, **kwargs)
        y_pred_test = model.predict(td.X_test_single, **kwargs)
        if td.X_val_single is not None:
            y_pred_val = model.predict(td.X_val_single, **kwargs)
        else:
            y_pred_val = None
        td.add_predictions(y_pred_train, y_pred_val, y_pred_test)

        td.add_metrics(Metrics(td))
        # Error over the horizon of a recursive application, as in an MPC (if td.eval_horizon is set)
        from physXAI.evaluation.rollout import evaluate_horizon
        evaluate_horizon(model, td)

    def _check_training_data(self, td: TrainingDataGeneric):
        if isinstance(td, TrainingDataMultiStep):
            raise TypeError(
                f"{self.__class__.__name__} is a single step model and needs PreprocessingSingleStep data. "
                "To evaluate it over a horizon, use PreprocessingSingleStep(eval_horizon=...)."
            )

    @classmethod
    def from_config(cls, config: dict) -> 'SingleStepModel':
        return cls(**config)


@register_model
class LinearRegressionModel(SingleStepModel):
    """
    A concrete implementation of SingleStepModel for scikit-learn's Linear Regression.
    """

    plot_title = "Linear Regression"

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

        if logger.isEnabledFor(logging.INFO):
            weights = "\n".join(f"  {inp}: {model.coef_[0][i]:.6g}" for i, inp in enumerate(td.columns))
            logger.info("Linear regression bias: %.6g, weights:\n%s", model.intercept_[0], weights)

    def compile_model(self, model):
        """
        No compilation step is needed for scikit-learn models.
        """
        pass

    def save_model(self, model, save_path: str):
        """
        Saves the trained LinearRegression model using joblib.

        Args:
            model (LinearRegression): The trained scikit-learn model.
            save_path (str): The path to save the model.
        """

        dump(model, output_path(save_path, '.joblib'))

    def load_model(self, load_path: str):
        """
        Loads a scikit-learn LinearRegression model from a file using joblib.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            LinearRegression: The loaded scikit-learn model.
        """

        model = joblib.load(input_path(load_path))
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
    def evaluate(model, td: TrainingDataMultiStep):
        """
        Predicts all sequences (with warmup, closed loop for fed back features) and stores the
        predicted trajectories, the metrics and the errors over the evaluation horizon in `td`.

        Args:
            model: The trained model (sequence or step model).
            td (TrainingDataMultiStep): The training data.
        """
        from physXAI.models.ann.model_construction.rnn_models import evaluate_sequences
        evaluate_sequences(model, td)

    def _check_training_data(self, td: TrainingDataGeneric):
        if not isinstance(td, TrainingDataMultiStep):
            raise TypeError(f"{self.__class__.__name__} is a multi step model and needs PreprocessingMultiStep data.")

    @classmethod
    def from_config(cls, config: dict) -> 'MultiStepModel':
        return cls(**config)
