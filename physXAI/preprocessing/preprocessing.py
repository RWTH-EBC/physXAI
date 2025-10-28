import os
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from physXAI.preprocessing.constructed import FeatureConstruction
from physXAI.preprocessing.training_data import TrainingData, TrainingDataMultiStep, TrainingDataGeneric
from physXAI.utils.logging import get_full_path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class PreprocessingData(ABC):
    """
    Abstract Preprocessing Class
    """

    def __init__(self, inputs: list[str], output: Union[str, list[str]], shift: int = 1,
                 time_step: Optional[Union[int, float]] = None,
                 test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42,
                 time_index_col: Union[str, float] = 0, csv_delimiter: str = ';', csv_encoding: str = 'latin1',
                 csv_header: int = 0, csv_skiprows: Union[int, list[int]] = [],):
        """
        Initializes the Preprocessing instance.

        Args:
            inputs (List[str]): List of column names to be used as input features.
            output (Union[str, List[str]]): Column name(s) for the target variable(s).
            shift (int): The number of time steps to shift the target variable for forecasting.
                         A shift of one means predicting the next time step.
            time_step (Optional[Union[int, float]]): Optional time step sampling. If None, sampling of data is used.
            test_size (float): Proportion of the dataset to allocate to the test set.
            val_size (float): Proportion of the dataset to allocate to the validation set.
            random_state (int): Seed for random number generators to ensure reproducible splits.
            time_index_col (Union[str, float]): Optional name or index of the time index column.
            csv_delimiter (str): Delimiter for csv data. Default is ';'.
            csv_encoding (str): Encoding for csv data. Default is 'latin1'.
            csv_header (int): Row number of csv header. Default is 0.
            csv_skiprows (Union[int, list[int]]): Row numbers of skipped data in csv. Default is no skipping.
        """
        self.time_index_col = time_index_col
        self.csv_delimiter = csv_delimiter
        self.csv_encoding = csv_encoding
        self.csv_header = csv_header
        self.csv_skiprows = csv_skiprows

        self.inputs: list[str] = inputs
        if isinstance(output, str):
            output = [output]
        self.output: list[str] = output
        self.shift: int = shift
        self.time_step = time_step

        # Training, validation and test size should be equal to 1
        assert test_size + val_size < 1
        self.test_size: float = test_size
        self.val_size: float = val_size

        self.random_state: int = random_state

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file into a Pandas DataFrame. CSV file should have a delimiter ';'

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """

        file_path = get_full_path(file_path)
        df = pd.read_csv(file_path,
                         delimiter=self.csv_delimiter,
                         index_col=self.time_index_col,
                         encoding=self.csv_encoding,
                         header=self.csv_header,
                         skiprows=self.csv_skiprows)

        # Determine Sampling
        sampling = df.index.to_series().diff().dropna().unique()
        assert len(sampling) == 1, f"Data Error: Training Data has different sampling times: {sampling}"
        time_step = sampling[0]
        if self.time_step is None:
            self.time_step = time_step
        else:
            assert self.time_step % time_step == 0, (f"Value Error: Given time step {self.time_step} is not a multiple "
                                                     f"of data time step: {time_step}.")
            filtering = (df.index - df.index[0]) % self.time_step == 0
            df = df[filtering]

        return df

    @abstractmethod
    def pipeline(self, file_path: str) -> TrainingDataGeneric:
        """
                Executes the full preprocessing pipeline.

                Args:
                    file_path (str): Path to the raw data CSV file.

                Returns:
                    TrainingDataGeneric: Preprocessed data ready for (multi-step) model training.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> 'PreprocessingData':
        pass


class PreprocessingSingleStep(PreprocessingData):
    """
    Handles preprocessing for single-step forecasting models.
    This includes loading data, applying feature constructions, selecting input/output columns,
    shifting the target variable for forecasting, and splitting data into training,
    validation, and test sets.
    """

    def __init__(self, inputs: list[str], output: Union[str, list[str]], shift: int = 1,
                 time_step: Optional[Union[int, float]] = None,
                 test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42,
                 time_index_col: Union[str, float] = 0, csv_delimiter: str = ';', csv_encoding: str = 'latin1',
                 csv_header: int = 0, csv_skiprows: Union[int, list[int]] = [], **kwargs):
        """
        Initializes the PreprocessingSingleStep instance.

        Args:
            inputs (List[str]): List of column names to be used as input features.
            output (Union[str, List[str]]): Column name(s) for the target variable(s).
            shift (int): The number of time steps to shift the target variable for forecasting.
                         A shift of one means predicting the next time step.
            time_step (Optional[Union[int, float]]): Optional time step sampling. If None, sampling of data is used.
            test_size (float): Proportion of the dataset to allocate to the test set.
            val_size (float): Proportion of the dataset to allocate to the validation set.
            random_state (int): Seed for random number generators to ensure reproducible splits.
            time_index_col (Union[str, float]): Optional name or index of the time index column.
            csv_delimiter (str): Delimiter for csv data. Default is ';'.
            csv_encoding (str): Encoding for csv data. Default is 'latin1'.
            csv_header (int): Row number of csv header. Default is 0.
            csv_skiprows (Union[int, list[int]]): Row numbers of skipped data in csv. Default is no skipping.
        """

        super().__init__(inputs, output, shift, time_step, test_size, val_size, random_state, time_index_col,
                         csv_delimiter, csv_encoding, csv_header, csv_skiprows)

    def process_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
                Processes the loaded DataFrame:
                1. Applies feature constructions defined in `FeatureConstruction`.
                2. Selects relevant input and output columns.
                3. Handles missing values by dropping rows.
                4. Shifts the target variable(s) `y` for forecasting.

                Args:
                    df (pd.DataFrame): The input DataFrame.

                Returns:
                    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the processed features (X)
                                                       and target (y) DataFrames.
        """

        # Applies feature constructions defined in `FeatureConstruction`.
        FeatureConstruction.process(df)

        df = df[self.inputs + [out for out in self.output if out not in self.inputs]]

        # Nan handling
        non_nan_rows = df.notna().all(axis=1)
        first_valid_index = non_nan_rows.idxmax() if non_nan_rows.any() else None
        last_valid_index = non_nan_rows.iloc[::-1].idxmax() if non_nan_rows.any() else None
        df = df.loc[first_valid_index:last_valid_index]
        if df.isnull().values.any():
            raise ValueError("Data Error: The TrainingData contains NaN values in intermediate rows.")

        X = df[self.inputs]
        y = df[self.output].shift(-self.shift)
        if self.shift > 0:  # pragma: no cover
            y = y.iloc[:-self.shift]
            X = X.iloc[:-self.shift]

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.DataFrame) -> TrainingData:
        """
        Splits the processed features (X) and target (y) into training, validation, and test sets.
        The data is converted to NumPy arrays before being stored in a TrainingData object.

        Args:
            X (pd.DataFrame): The DataFrame of input features.
            y (pd.DataFrame): The DataFrame of target variables.

        Returns:
            TrainingData: A data container with NumPy arrays for X_train, X_val, X_test,
                          y_train, y_val, y_test, and original column names.
        """

        # Train - Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(self.test_size + self.val_size),
                                                            random_state=self.random_state)

        # Validation - Test split
        split = self.test_size / (self.test_size + self.val_size)
        if self.val_size != 0:
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=split,
                                                            random_state=self.random_state)
            X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
        else:
            X_val, y_val = None, None

        # Return TrainingData object
        return TrainingData(X_train.to_numpy(), X_val, X_test.to_numpy(),
                            y_train.to_numpy(), y_val, y_test.to_numpy(),
                            X.columns.values.tolist())

    def pipeline(self, file_path: str) -> TrainingData:
        """
        Executes the full preprocessing pipeline: load, process, and split data.

        Args:
            file_path (str): Path to the raw data CSV file.

        Returns:
            TrainingData: The preprocessed data ready for model training.
        """

        df = self.load_data(file_path)
        X, y = self.process_data(df)
        td = self.split_data(X, y)
        td.add_file_path(file_path)
        return td

    def get_config(self) -> dict:
        config = {
            '__class_name__': self.__class__.__name__,
            'inputs': self.inputs,
            'output': self.output,
            'shift': self.shift,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'time_step': self.time_step,
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'PreprocessingSingleStep':
        return cls(**config)


class PreprocessingMultiStep (PreprocessingData):
    """
    Handles preprocessing for multi-step forecasting models, typically RNNs.
    This involves creating windowed datasets suitable for sequence models,
    including optional warmup sequences.
    """

    def __init__(self, inputs: list[str], output: Union[str, list[str]], label_width: int,  warmup_width: int, shift: int = 1,
                 time_step: Optional[Union[int, float]] = None,
                 test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42,
                 time_index_col: Union[str, float] = 0, csv_delimiter: str = ';', csv_encoding: str = 'latin1',
                 csv_header: int = 0, csv_skiprows: Union[int, list[int]] = [],
                 overlapping_sequences: bool = True, batch_size=32, init_features: list[str] = None,**kwargs):
        """
        Initializes the PreprocessingMultiStep instance.

        Args:
            inputs (List[str]): Column names for input features to the main RNN.
            output (Union[str, List[str]]): Column name(s) for target variable(s).
            label_width (int): Number of time steps in the output (label) sequence.
            warmup_width (int): Number of time steps in the warmup sequence (for RNN state initialization).
                                If 0, no warmup sequence is used.
            shift (int): Offset between the end of the input window and the start of the label window.
            time_step (Optional[Union[int, float]]): Optional time step sampling. If None, sampling of data is used.
            test_size (float): Proportion for the test set.
            val_size (float): Proportion for the validation set.
            random_state (int): Seed for reproducibility (though shuffle in timeseries_dataset_from_array
                                might behave differently with seeds across calls if not reset).
            time_index_col (Union[str, float]): Optional name or index of the time index column.
            csv_delimiter (str): Delimiter for csv data. Default is ';'.
            csv_encoding (str): Encoding for csv data. Default is 'latin1'.
            csv_header (int): Row number of csv header. Default is 0.
            csv_skiprows (Union[int, list[int]]): Row numbers of skipped data in csv. Default is no skipping.
            overlapping_sequences (bool): Whether to use overlapping sequences to generate multi-step sequences.
            batch_size (int): Batch size for creating tf.data.Dataset objects.
            init_features (Optional[List[str]]): Features to include in the warmup sequence.
                                                 If None and warmup_width > 0, defaults to `inputs`.
                                                 If None and warmup_width <= 0, defaults to empty list.
        """
        super().__init__(inputs, output, shift, time_step, test_size, val_size, random_state, time_index_col,
                         csv_delimiter, csv_encoding, csv_header, csv_skiprows)

        self.overlapping_sequences = overlapping_sequences

        # Determine initialization features
        if init_features is None:
            self.init_features: list[str] = self.output
        else:
            self.init_features: list[str] = init_features

        keras.utils.set_random_seed(random_state)

        # Determine necessary parameters for window creation
        self.features: list[str] = (inputs + self.output +
                                    [f for f in self.init_features if f not in inputs and f not in self.output])
        self.column_indices: dict[str, int] = {name: i for i, name in enumerate(self.features)}
        self.warmup_columns_input: list[str] = list(set(self.init_features) & set(inputs))
        self.warmup_columns_labels: list[str] = list(set(self.init_features) - set(inputs))

        self.label_width: int = label_width
        self.warmup_width: int = warmup_width
        self.shift: int = shift
        self.total_window_size: int = warmup_width + label_width + shift

        self.input_end: int = warmup_width + label_width
        self.label_start: int = shift + warmup_width

        self.input_slice = slice(self.warmup_width, self.input_end)
        self.labels_slice = slice(self.label_start, None)
        self.warmup_slice_input = slice(0, self.warmup_width)
        self.warmup_slice_labels = slice(shift, self.label_start)

        self.batch_size: int = batch_size

    def process_data(self, df: pd.DataFrame) -> TrainingDataMultiStep:
        """
        Processes the loaded DataFrame for multi-step forecasting:
        1. Applies feature constructions defined in `FeatureConstruction`.
        2. Selects relevant features.
        3. Handles missing values by dropping rows.
        4. Creates windowed tf.data.Dataset objects for training, validation, and testing.
        5. Maps these datasets to split windows into (inputs, warmup_sequence), labels structure.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            TrainingDataMultiStep: Container with tf.data.Dataset objects.
        """

        # Applies feature constructions defined in `FeatureConstruction`.
        FeatureConstruction.process(df)

        df = df[self.features]
        non_nan_rows = df.notna().all(axis=1)
        first_valid_index = non_nan_rows.idxmax() if non_nan_rows.any() else None
        last_valid_index = non_nan_rows.iloc[::-1].idxmax() if non_nan_rows.any() else None
        df = df.loc[first_valid_index:last_valid_index]
        if df.isnull().values.any():
            raise ValueError("Data Error: The TrainingData contains NaN values in intermediate rows.")

        # Create windowed dataset
        train_ds, val_ds, test_ds = self._make_dataset(df)

        # Split data
        train_ds = train_ds.map(lambda x: self._split_window(x))
        if val_ds is not None:
            val_ds = val_ds.map(lambda x: self._split_window(x))
        test_ds = test_ds.map(lambda x: self._split_window(x))

        return TrainingDataMultiStep(train_ds, val_ds, test_ds, self.inputs, self.output, self.init_features)

    def _split_window(self, features):   # pragma: no cover
        """
        Splits a batch of windows into inputs, (optional) warmup, and labels.
        This function is designed to be used with `tf.data.Dataset.map()`.

        Args:
            features (tf.Tensor): A batch of windows from `timeseries_dataset_from_array`.
                                      Shape: (batch_size, total_window_size, num_all_df_features).

        Returns:
            Tuple: Depending on `warmup_width`:
                   - If `warmup_width > 0`: `((main_inputs, warmup_inputs), labels)`
                   - If `warmup_width == 0`: `(main_inputs, labels)`
        """

        # Get inputs
        inputs = keras.ops.stack([features[:, self.input_slice, self.column_indices[name]]
                                  for name in self.inputs], axis=-1)

        # Get labels
        labels = keras.ops.stack([features[:, self.labels_slice, self.column_indices[name]]
                                  for name in self.output], axis=-1)

        # Warmup
        if self.warmup_width > 0:
            if len(self.warmup_columns_input) > 0 and len(self.warmup_columns_labels) > 0:
                # Get warmup features from labels and features
                warmup_inputs = keras.ops.stack([features[:, self.warmup_slice_input, self.column_indices[name]]
                                                 for name in self.warmup_columns_input], axis=-1)
                warmup_labels = keras.ops.stack([features[:, self.warmup_slice_labels, self.column_indices[name]]
                                                 for name in self.warmup_columns_labels], axis=-1)
                warmup = keras.ops.concatenate([warmup_inputs, warmup_labels], axis=-1)
            elif len(self.warmup_columns_input) > 0:
                # Get warmup features from features
                warmup = keras.ops.stack([features[:, self.warmup_slice_input, self.column_indices[name]]
                                          for name in self.warmup_columns_input], axis=-1)
            else:
                # Get warmup features from labels
                warmup = keras.ops.stack([features[:, self.warmup_slice_labels, self.column_indices[name]]
                                          for name in self.warmup_columns_labels], axis=-1)
            return (inputs, warmup), labels
        else:
            return inputs, labels

    def _make_dataset(self, df: pd.DataFrame):
        """
        Creates windowed tf.data.Dataset objects for training, validation, and testing
        from the input DataFrame.

        Args:
            df (pd.DataFrame): The processed DataFrame containing all necessary features.

        Returns:
            Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.data.Dataset]:
                Train, validation (or None), and test datasets.
        """

        data = np.array(df, dtype=np.float32)

        if self.overlapping_sequences:
            sequence_stride = 1
        else:
            sequence_stride = self.label_width

        # et a batch of sequences
        ds = keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride,
            shuffle=True,
            batch_size=self.batch_size)

        # Split the dataset into training and temporary batch_dataset
        total_batches = len(ds)
        train_batches = int(total_batches * (1 - self.val_size - self.test_size))
        train_ds = ds.take(train_batches)
        temp_ds = ds.skip(train_batches)

        # Split the dataset into validation and test batch_dataset
        val_ratio_temp = self.val_size / (self.val_size + self.test_size)
        total_batches = len(temp_ds)
        train_batches = int(total_batches * val_ratio_temp)
        val_ds = temp_ds.take(train_batches)
        test_ds = temp_ds.skip(train_batches)

        if len(val_ds) == 0:
            val_ds = None

        return train_ds, val_ds, test_ds

    def pipeline(self, file_path: str) -> TrainingDataMultiStep:
        """
        Executes the full multi-step preprocessing pipeline.

        Args:
            file_path (str): Path to the raw data CSV file.

        Returns:
            TrainingDataMultiStep: Preprocessed data ready for multi-step model training.
        """

        df = self.load_data(file_path)
        td = self.process_data(df)
        td.add_file_path(file_path)
        return td

    def get_config(self) -> dict:
        config = {
            '__class_name__': self.__class__.__name__,
            'inputs': self.inputs,
            'output': self.output,
            'label_width': self.label_width,
            'warmup_width': self.warmup_width,
            'overlapping_sequences': self.overlapping_sequences,
            'batch_size': self.batch_size,
            'init_features': self.init_features,
            'shift': self.shift,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'time_step': self.time_step,
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'PreprocessingMultiStep':
        return cls(**config)
