import os
from abc import ABC, abstractmethod
from typing import Optional, Union, Iterable
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from physXAI.preprocessing.constructed import FeatureConstruction, FeatureBase
from physXAI.preprocessing.training_data import TrainingData, TrainingDataMultiStep, TrainingDataGeneric
from physXAI.utils.logging import get_full_path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def convert_shift_to_dict(s: Union[int, str, dict], inputs: list[str], custom_default: Union[int, str] = None) -> dict:
    """
    Convert a given shift variable (int, str) into a dictionary in which a shift is defined for every input.
    If a dictionary is given as shift, check entries and autocomplete dict if necessary.

    Args:
        s (Union[int, str, dict]): Shift value. Either a single string or int which then will be applied to all the inputs or
            a dictionary in which a different shift can be defined for each input. If the dictionary does not specify the
            shift for all inputs, the shift for inputs not specified is set to the default value (autocomplete)
        inputs (list(str)): List of Input variables
        custom_default (Union[int, str]): if no custom default is specified, 'previous' is used as default shift
    """

    def return_valid_shift(val: Union[int, str]):
        """ check the validity of the given shift and return a string if val is int """
        if val in ['current', 0]:
            val = 'current'
        elif val in ['previous', 1]:
            val = 'previous'
        elif val == 'mean_over_interval':
            val = 'mean_over_interval'
        else:
            raise ValueError(
                f"Value of shift not supported, value is: {val}. Shift must be 'current' (or 0 if s is int), "
                f"'previous' (or 1 if s is int) or 'mean_over_interval'.")
        return val

    # set custom default or - if no custom default is specified - use 'previous' as default
    default = 'previous' if custom_default is None else return_valid_shift(custom_default)

    if isinstance(s, (int, str)):
        d = {}
        s = return_valid_shift(s)

        # add shift for each input
        for inp in inputs:
            d.update({inp: s})
        return d

    elif isinstance(s, dict):
        def get_lag(inputs: list[str], current_input: str) -> int:
            """ get lag of current input """
            count = 0
            for inp in inputs:
                spl = inp.split(current_input) # make sure it is the current input
                if spl[0] == '' and spl[1] != '' and spl[1].split('_lag')[0] == '':
                    count += 1
            return count

        # check if lags exist
        d = {}
        inputs_without_lags = {}
        for inp in inputs:
            # skip if current input is just the lag of another inp
            if not inp.__contains__('_lag'):
                inputs_without_lags.update({inp: get_lag(inputs, inp)})

        for inp in inputs_without_lags.keys():
            # if an input has a shift assigned already, the validity is checked
            # otherwise default value is assigned
            if inp in s.keys():
                d.update({inp: return_valid_shift(s[inp])})
            else:
                d.update({inp: default})

            # all inputs with lags should have the same shift
            if inputs_without_lags[inp] > 0: # if current input has lags
                for i in range(inputs_without_lags[inp]):
                    name = inp + '_lag' + str(i+1)

                    # if a shift was already defined for this lag, check if it matches the shift of the original inp
                    if name in s.keys():
                        assert return_valid_shift(s[name]) == d[inp], \
                            'Make sure that all lags of an input have the same shift'
                    d.update({name: d[inp]})
        return d
    else:
        raise TypeError(f'shift must be of type int, str or dict, is type {type(s)}')


class PreprocessingData(ABC):
    """
    Abstract Preprocessing Class
    """

    def __init__(self, inputs: list[Union[str, FeatureBase]], output: Union[str, list[str]],
                 time_step: Optional[Union[int, float]] = None,
                 test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42,
                 time_index_col: Union[str, float] = 0, csv_delimiter: str = ';', csv_encoding: str = 'latin1',
                 csv_header: int = 0, csv_skiprows: Union[int, list[int]] = [], ignore_nan: bool = False, **kwargs):
        """
        Initializes the Preprocessing instance.

        Args:
            inputs (List[Union[str, FeatureBase]]): List of column names or Features to be used as input features.
            output (Union[str, List[str]]): Column name(s) for the target variable(s).
            time_step (Optional[Union[int, float]]): Optional time step sampling. If None, sampling of data is used.
            test_size (float): Proportion of the dataset to allocate to the test set.
            val_size (float): Proportion of the dataset to allocate to the validation set.
            random_state (int): Seed for random number generators to ensure reproducible splits.
            time_index_col (Union[str, float]): Optional name or index of the time index column.
            csv_delimiter (str): Delimiter for csv data. Default is ';'.
            csv_encoding (str): Encoding for csv data. Default is 'latin1'.
            csv_header (int): Row number of csv header. Default is 0.
            csv_skiprows (Union[int, list[int]]): Row numbers of skipped data in csv. Default is no skipping.
            ignore_nan (bool): If True, rows with NaN values will be dropped. If False, an error is raised if NaNs are present. Default is False.
        """
        self.time_index_col = time_index_col
        self.csv_delimiter = csv_delimiter
        self.csv_encoding = csv_encoding
        self.csv_header = csv_header
        self.csv_skiprows = csv_skiprows

        self.inputs: list[str] = FeatureConstruction.process_inputs(inputs)
        if isinstance(output, str):
            output = [output]
        self.output: list[str] = output

        self.time_step = time_step

        # Training, validation and test size should be equal to 1
        assert test_size + val_size < 1
        self.test_size: float = test_size
        self.val_size: float = val_size

        self.random_state: int = random_state

        self.ignore_nan: bool = ignore_nan

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

        return df

    def sample_df_according_to_timestep(self, df: pd.DataFrame):
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

    def __init__(self, inputs: list[Union[str, FeatureBase]], output: Union[str, list[str]],
                 time_step: Optional[Union[int, float]] = None,
                 test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42,
                 time_index_col: Union[str, float] = 0, csv_delimiter: str = ';', csv_encoding: str = 'latin1',
                 csv_header: int = 0, csv_skiprows: Union[int, list[int]] = [], ignore_nan: bool = False, **kwargs):
        """
        Initializes the PreprocessingSingleStep instance.

        Args:
            inputs (List[Union[str, FeatureBase]]): List of column names or Features to be used as input features.
            output (Union[str, List[str]]): Column name(s) for the target variable(s).
            time_step (Optional[Union[int, float]]): Optional time step sampling. If None, sampling of data is used.
            test_size (float): Proportion of the dataset to allocate to the test set.
            val_size (float): Proportion of the dataset to allocate to the validation set.
            random_state (int): Seed for random number generators to ensure reproducible splits.
            time_index_col (Union[str, float]): Optional name or index of the time index column.
            csv_delimiter (str): Delimiter for csv data. Default is ';'.
            csv_encoding (str): Encoding for csv data. Default is 'latin1'.
            csv_header (int): Row number of csv header. Default is 0.
            csv_skiprows (Union[int, list[int]]): Row numbers of skipped data in csv. Default is no skipping.
            ignore_nan (bool): If True, rows with NaN values will be dropped. If False, an error is raised if NaNs are present. Default is False.
        """

        if 'shift' in kwargs.keys():
            DeprecationWarning(
                "shift parameter is deprecated for SingleStep models and replaced by sampling_method, an attribute of "
                "each Feature. This allows specifying individual 'shifts' for each Feature / input. A default sampling"
                "method can be specified via FeatureConstruction.set_default_sampling_method(<your default sampling>)."
            )
            DeprecationWarning(
                f"shift parameter was given as shift={kwargs['shift']}. Setting FeatureConstruction.set_default_"
                f"sampling_method(shift) and override possible individual sampling methods of all Features. If this is"
                f"not intended, remove shift parameter when initializing PreprocessingSingleStep object!"
            )
            FeatureConstruction.set_default_sampling_method(kwargs['shift'])
            for f in FeatureConstruction.features:
                f.sampling_method = kwargs['shift']

        super().__init__(inputs, output, time_step, test_size, val_size, random_state, time_index_col,
                         csv_delimiter, csv_encoding, csv_header, csv_skiprows, ignore_nan, **kwargs)

    def process_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
                Processes the loaded DataFrame:
                1. Applies feature constructions defined in `FeatureConstruction`.
                2. Selects relevant input and output columns.
                3. Handles missing values by dropping rows.
                4. Applies the defined sampling method on each input variable.

                Args:
                    df (pd.DataFrame): The input DataFrame.

                Returns:
                    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the processed features (X)
                                                       and target (y) DataFrames.
        """

        # extract the names of all features in inputs and outputs that are based on lagged features
        lag_based_features = FeatureConstruction.get_features_including_lagged_features(self.inputs + self.output)

        inputs_without_lags = [inp for inp in self.inputs if inp not in lag_based_features]

        # Applies feature constructions defined in `FeatureConstruction`.
        # Only apply for those features that are not lags since lags must be constructed after sampling the data
        # according to the given time step
        FeatureConstruction.process(df, feature_names=inputs_without_lags + [out for out in self.output if out not in inputs_without_lags])
        features_without_lags: list[FeatureBase] = [FeatureConstruction.get_feature(inp) for inp in inputs_without_lags]

        df = df[inputs_without_lags + [out for out in self.output if out not in inputs_without_lags]]

        # Nan handling in first and last rows
        non_nan_rows = df.notna().all(axis=1)
        first_valid_index = non_nan_rows.idxmax() if non_nan_rows.any() else None
        last_valid_index = non_nan_rows.iloc[::-1].idxmax() if non_nan_rows.any() else None
        df = df.loc[first_valid_index:last_valid_index]

        def get_mean_over_interval(y: pd.DataFrame, x: pd.DataFrame):
            """return mean values of x on target sampling (index of y)"""
            def pairwise(iterable: Iterable):
                "s -> (s0,s1), (s1,s2), (s2, s3), ..."
                a, b = itertools.tee(iterable)
                next(b, None)
                return zip(a, b)

            original_grid = np.array(x.index)
            results = []
            for i, j in pairwise(y.index): # output interval is target grid
                slicer = np.logical_and(original_grid >= i, original_grid < j)
                d = {'Index': j}
                for inp in x.columns:
                    d[inp] = x[inp][slicer].mean()
                results.append(d)

            x = pd.DataFrame(results).set_index('Index')

            return x

        # output is independent of sampling of inputs -> sample according to time step already
        y = df[self.output].copy()
        y = self.sample_df_according_to_timestep(y)

        X = df[inputs_without_lags].copy()

        if all('current' == f.sampling_method for f in features_without_lags):
            # filter / sample data
            X = self.sample_df_according_to_timestep(X)
            # nothing more to do here
        elif all('previous' == f.sampling_method for f in features_without_lags):
            # filter / sample data
            X = self.sample_df_according_to_timestep(X)

            # shift data by 1 and shorten DataFrames accordingly
            X = X.shift(1)
            y = y.iloc[1:]
            X = X.iloc[1:]
        elif all('mean_over_interval' == f.sampling_method for f in features_without_lags):
            X = get_mean_over_interval(y, X)
            # synchronize length between X and y
            y = y.iloc[1:]

        else:  # different inputs have different sampling methods
            res = []
            previous_or_mean_in_sampling_methods = False
            for f in features_without_lags:
                # only process inputs with sampling method mean_over_interval first since X cannot be sampled
                # to the actual required time steps until the intermediate values were taken into the mean
                if f.sampling_method == 'mean_over_interval':
                    res.append(get_mean_over_interval(y, X[[f.feature]]))
                    previous_or_mean_in_sampling_methods = True

            # sample X according to required time step
            X = self.sample_df_according_to_timestep(X)
            # process inputs with sampling methods 'current' and 'previous'
            for f in features_without_lags:
                _x = X[[f.feature]]
                if f.sampling_method == 'current':
                    # no transformation needed
                    res.append(_x)
                elif f.sampling_method == 'previous':
                    # shift by 1
                    _x = _x.shift(1)
                    _x = _x.iloc[1:]
                    res.append(_x)
                    previous_or_mean_in_sampling_methods = True
                elif f.sampling_method == 'mean_over_interval':
                    continue
                else:
                    raise NotImplementedError(f"Sampling method '{f.sampling_method}' not implemented.")

            X = pd.concat(res, axis=1)

            # Sampling methods 'previous' and 'mean_over_interval' reduce available data points by 1.
            # Therefore, lengths of X and y have to be synchronized
            if previous_or_mean_in_sampling_methods:
                y = y.iloc[1:]
                X = X.sort_index(ascending=True)
                X = X.iloc[1:]

        res_df = pd.concat([X, y], axis=1)

        if res_df.isnull().values.any():
            if self.ignore_nan:
                res_df.dropna(inplace=True)
            else:
                raise ValueError(
                    "Data Error: The TrainingData contains NaN values in intermediate rows. If this is intended, set "
                    "ignore_nan=True in PreprocessingSingleStep.")

        # Applies feature constructions defined in `FeatureConstruction` to the lagged inputs
        FeatureConstruction.process(res_df, feature_names=lag_based_features)

        # drop NaNs occurring due to creation of lags
        res_df.dropna(inplace=True)

        X = res_df[self.inputs]
        y = res_df[self.output]

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
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'time_step': self.time_step,
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'PreprocessingSingleStep':
        return cls(**config)


class PreprocessingMultiStep(PreprocessingData):
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
                 overlapping_sequences: bool = True, batch_size=32, init_features: list[str] = None, **kwargs):
        """
        Initializes the PreprocessingMultiStep instance.

        Args:
           inputs (List[Union[str, FeatureBase]]): List of column names or Features that are input features to the main RNN.
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
        super().__init__(inputs, output, time_step, test_size, val_size, random_state, time_index_col,
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

        # filter data
        df = self.sample_df_according_to_timestep(df)

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
