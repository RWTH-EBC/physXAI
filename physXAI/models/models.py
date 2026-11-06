import os
import time
from abc import ABC, abstractmethod
from typing import Type
import joblib
from joblib import dump
from sklearn.linear_model import LinearRegression
from physXAI.utils.logging import get_full_path, create_full_path, Logger
from physXAI.preprocessing.training_data import TrainingData, TrainingDataMultiStep, TrainingDataGeneric
from physXAI.evaluation.metrics import Metrics
from physXAI.plotting.plotting import model_report, save_figure
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


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
        Saves the figure of `plot` as html file next to the model: `<save_path>.html`, or in the run
        directory of the Logger if no path is given. Nothing is saved without any of both.
        """
        if save_path is None:
            if Logger._logger is None:
                return None
            save_path = Logger.get_model_savepath()
        stem = os.path.splitext(save_path)[0] if save_path.endswith((".keras", ".joblib")) else save_path
        return save_figure(figure, create_full_path(stem + ".html"))

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

    def prepare(self):
        """Hook called before the model is generated (e.g. to set random seeds)."""

    def _check_training_data(self, td: TrainingDataGeneric):
        """Hook checking that the training data fits the model, called before the model is generated."""

    def pipeline(self, td: TrainingDataGeneric, save_path: str = None, plot: bool = True, save_model: bool = True,
                 show_plot: bool = False):
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
              plot (bool, optional): Whether to plot the results (saved as html next to the model). Defaults to True.
              save_model (bool, optional): Whether to save the trained model. Defaults to True.
              show_plot (bool, optional): Whether to open the plot. Defaults to False.

          Returns:
              The trained model instance.
          """

        self._check_training_data(td)
        self.prepare()
        model = self.generate_model(td=td)
        self.compile_model(model)
        self.fit_model(model, td)

        self.evaluate(model, td)
        if plot:
            self.save_plot(self.plot(td, show=show_plot), save_path)
        if save_model:
            self.save_model(model, save_path=save_path)

        return model

    def online_pipeline(self, td: TrainingDataGeneric, load_path: str, save_path: str = None,
                        plot: bool = True, save_model: bool = True, show_plot: bool = False):
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

        self._check_training_data(td)
        model = self.load_model(load_path)
        self.compile_model(model)

        model = self.online_pipeline_internal(td, model)

        if plot:
            self.save_plot(self.plot(td, show=show_plot), save_path)
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
        Logger.print(f"Warning: Class '{cls.__name__}' is already registered. Overwriting.", 'warning')
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
            kwargs['verbose'] = Logger.verbosity()

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

        if Logger.check_print_level('INFO'):
            print("Trained Linear Regression Coefficients:")
            print("Bias:", model.intercept_[0])
            for i, inp in enumerate(td.columns):
                print(f"Weight for {inp}:", model.coef_[0][i])

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
