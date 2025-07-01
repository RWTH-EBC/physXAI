from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class TrainingDataGeneric(ABC):
    """
     A generic container class to hold all data related to a machine learning model's lifecycle.
    This includes training, validation, and test datasets (as NumPy arrays),
    model predictions, evaluation metrics, training history, and training time.
    """

    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.y_train_pred = None
        self.y_val_pred = None
        self.y_test_pred = None

        self.file_path = None

        self.training_record = None

        self.training_time = None

        self.metrics = None

        self.columns = None

    def add_training_record(self, data):
        """
        Stores the training history or record.
        For Keras models, this is typically the `History` object returned by `model.fit()`.

        Args:
            data: The training record data.
        """
        self.training_record = data

    def add_predictions(self, y_train_pred: np.array, y_val_pred: np.array, y_test_pred: np.array):
        """
        Stores the model's predictions for the training, validation, and test sets.

        Args:
            y_train_pred (np.ndarray): Predictions on the training set.
            y_val_pred (Optional[np.ndarray]): Predictions on the validation set.
            y_test_pred (np.ndarray): Predictions on the test set.
        """

        self.y_train_pred = y_train_pred
        self.y_val_pred = y_val_pred
        self.y_test_pred = y_test_pred

    def add_metrics(self, metrics):
        """
        Stores the calculated evaluation metrics.

        Args:
            metrics: The metrics object.
        """

        self.metrics = metrics

    def add_training_time(self, time: float):
        """
        Stores the duration of the model training process.

        Args:
            time (float): The training time in seconds.
        """
        self.training_time = time

    def add_file_path(self, path: str):
        self.file_path = path

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @property
    @abstractmethod
    def X_train_single(self):
        pass

    @property
    @abstractmethod
    def y_train_single(self):
        pass

    @property
    @abstractmethod
    def X_val_single(self):
        pass

    @property
    @abstractmethod
    def y_val_single(self):
        pass

    @property
    @abstractmethod
    def X_test_single(self):
        pass

    @property
    @abstractmethod
    def y_test_single(self):
        pass

    @property
    @abstractmethod
    def y_train_pred_single(self):
        pass

    @property
    @abstractmethod
    def y_val_pred_single(self):
        pass

    @property
    @abstractmethod
    def y_test_pred_single(self):
        pass


class TrainingData(TrainingDataGeneric):
    """
    A container class to hold all data related to a single-step machine learning model's lifecycle.
    This includes training, validation, and test datasets (as NumPy arrays),
    model predictions, evaluation metrics, training history, and training time.
    """

    def __init__(self, X_train: np.array, X_val: np.array, X_test: np.array,
                 y_train: np.array, y_val: np.array, y_test: np.array,
                 columns: list[str]):
        """
        Initializes the TrainingData object.

        Args:
            X_train (np.ndarray): NumPy array of training features.
            X_val (Optional[np.ndarray]): NumPy array of validation features. Can be None.
            X_test (np.ndarray): NumPy array of test features.
            y_train (np.ndarray): NumPy array of training target values.
            y_val (Optional[np.ndarray]): NumPy array of validation target values. Can be None.
            y_test (np.ndarray): NumPy array of test target values.
            columns (List[str]): List of input feature names (columns of X).
        """
        super().__init__()

        self.X_train: np.ndarray = X_train
        self.X_val: np.ndarray = X_val
        self.X_test: np.ndarray = X_test
        self.y_train: np.ndarray = y_train
        self.y_val: np.ndarray = y_val
        self.y_test: np.ndarray = y_test
        self.columns: list[str] = columns

    def get_config(self) -> dict:
        config = {
            'file_path': self.file_path,
            'metrics': self.metrics.get_config() if self.metrics is not None else None,
            'training_time': self.training_time,
            'training_record': self.training_record.history if self.training_record is not None else None,
        }
        return config

    @property
    def X_train_single(self):
        return self.X_train

    @property
    def y_train_single(self):
        return self.y_train

    @property
    def X_val_single(self):
        return self.X_val

    @property
    def y_val_single(self):
        return self.y_val

    @property
    def X_test_single(self):
        return self.X_test

    @property
    def y_test_single(self):
        return self.y_test

    @property
    def y_train_pred_single(self):
        return self.y_train_pred

    @property
    def y_val_pred_single(self):
        return self.y_val_pred

    @property
    def y_test_pred_single(self):
        return self.y_test_pred


class TrainingDataMultiStep(TrainingDataGeneric):
    """
    A container class for data related to multi-step forecasting models,
    typically using tf.data.Dataset objects for handling windowed sequence data.
    It also extracts NumPy array versions of these datasets for easier inspection or
    use with libraries that expect NumPy arrays.
    """

    def __init__(self, train_ds, val_ds, test_ds, columns: list[str], output: list[str], init_columns: list[str]):
        """
        Initializes the TrainingDataMultiStep object.

        Args:
            train_ds (tf.data.Dataset): TensorFlow Dataset for training.
                                        Each element is typically a tuple (features, labels).
                                        Features can be a single tensor or a tuple (e.g., (main_input, warmup_input)).
            val_ds (Optional[tf.data.Dataset]): TensorFlow Dataset for validation. Can be None.
            test_ds (tf.data.Dataset): TensorFlow Dataset for testing.
            columns (List[str]): List of input feature names (columns of X).
            output  (str): (List of) Name(s) of the output column(s).
        """
        super().__init__()

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.columns = columns
        self.output = output
        self.init_columns = init_columns

        # Extract numpy objects
        self._Xy_train()
        if self.val_ds is not None:
            self._Xy_val()
        else:
            self.X_val = None
            self.y_val = None
        if self.test_ds is not None:
            self._Xy_test()
        else:
            self.X_test = None
            self.y_test = None

        self.single_step_metrics = None

    def add_single_step_metrics(self, metrics):
        """
        Stores the calculated evaluation single step metrics.

        Args:
            metrics: The metrics object.
        """

        self.single_step_metrics = metrics

    def _Xy_train(self):
        """
        Converts the training tf.data.Dataset into NumPy arrays.
        """
        y_values = []
        X_values = []
        X_values_init = []
        for X, y in self.train_ds:
            if isinstance(X, tuple):
                # With warmup
                X_values.append(X[0].numpy())
                X_values_init.append(X[1].numpy())
            else:
                # Without warmup
                X_values.append(X.numpy())
            y_values.append(y.numpy())
        self.X_train = np.concatenate(X_values)
        if len(X_values_init) > 0:
            self.X_train = (self.X_train, np.concatenate(X_values_init))
        self.y_train = np.concatenate(y_values)

    def _Xy_val(self):
        """
        Converts the validation tf.data.Dataset into NumPy arrays.
        """
        y_values = []
        X_values = []
        X_values_init = []
        for X, y in self.val_ds:
            if isinstance(X, tuple):
                X_values.append(X[0].numpy())
                X_values_init.append(X[1].numpy())
            else:
                X_values.append(X.numpy())
            y_values.append(y.numpy())
        self.X_val = np.concatenate(X_values)
        if len(X_values_init) > 0:
            self.X_val = (self.X_val, np.concatenate(X_values_init))
        self.y_val = np.concatenate(y_values)

    def _Xy_test(self):
        """
        Converts the test tf.data.Dataset into NumPy arrays.
        """
        y_values = []
        X_values = []
        X_values_init = []
        for X, y in self.test_ds:
            if isinstance(X, tuple):
                X_values.append(X[0].numpy())
                X_values_init.append(X[1].numpy())
            else:
                X_values.append(X.numpy())
            y_values.append(y.numpy())
        self.X_test = np.concatenate(X_values)
        if len(X_values_init) > 0:
            self.X_test = (self.X_test, np.concatenate(X_values_init))
        self.y_test = np.concatenate(y_values)

    def get_config(self) -> dict:
        config = {
            'file_path': self.file_path,
            'metrics': self.metrics.get_config() if self.metrics is not None else None,
            'training_time': self.training_time,
            'training_record': self.training_record.history if self.training_record is not None else None,
        }
        return config

    @property
    def X_train_single(self):
        if isinstance(self.X_train, tuple):
            X = self.X_train[0]
        else:
            X = self.X_train
        return X.reshape(-1, *X.shape[2:])

    @property
    def X_train_init(self):
        if isinstance(self.X_train, tuple):
            X = self.X_train[1]
        else:
            X = None
        return X

    @property
    def X_train_features(self):
        if isinstance(self.X_train, tuple):
            X = self.X_train[0]
        else:
            X = self.X_train
        return X

    @property
    def y_train_single(self):
        return self.y_train.reshape(-1, *self.y_train.shape[2:])

    @property
    def X_val_single(self):
        if isinstance(self.X_val, tuple):
            X = self.X_val[0]
        else:
            X = self.X_val
        return X.reshape(-1, *X.shape[2:]) if X is not None else None

    @property
    def X_val_init(self):
        if isinstance(self.X_val, tuple):
            X = self.X_val[1]
        else:
            X = None
        return X

    @property
    def X_val_features(self):
        if isinstance(self.X_val, tuple):
            X = self.X_val[0]
        else:
            X = self.X_val
        return X

    @property
    def y_val_single(self):
        return self.y_val.reshape(-1, *self.y_val.shape[2:]) if self.y_val is not None else None

    @property
    def X_test_single(self):
        if isinstance(self.X_test, tuple):
            X = self.X_test[0]
        else:
            X = self.X_test
        return X.reshape(-1, *X.shape[2:])

    @property
    def X_test_init(self):
        if isinstance(self.X_test, tuple):
            X = self.X_test[1]
        else:
            X = None
        return X

    @property
    def X_test_features(self):
        if isinstance(self.X_test, tuple):
            X = self.X_test[0]
        else:
            X = self.X_test
        return X

    @property
    def y_test_single(self):
        return self.y_test.reshape(-1, *self.y_test.shape[2:])

    @property
    def y_train_pred_single(self):
        if self.y_train_pred.ndim == 3:
            return self.y_train_pred.reshape(-1, *self.y_train_pred.shape[2:])
        else:
            return self.y_train_pred

    @property
    def y_val_pred_single(self):
        if self.y_val_pred.ndim == 3:
            return self.y_val_pred.reshape(-1, *self.y_val_pred.shape[2:]) if self.y_val_pred is not None else None
        else:
            return self.y_val_pred

    @property
    def y_test_pred_single(self):
        if self.y_test_pred.ndim == 3:
            return self.y_test_pred.reshape(-1, *self.y_test_pred.shape[2:])
        else:
            return self.y_test_pred
