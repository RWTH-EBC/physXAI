from typing import Union, Iterable
import pandas as pd
import numpy as np
import itertools


def _return_valid_sampling_method(v: Union[int, str]):
    """ check the validity of the given sampling method and return a string if value is int """

    if not isinstance(v, (int, str)):
        raise TypeError(f'Type of sampling method not supported. Type is {type(v)}, must be int or str.')

    if v in ['current', 0]:
        return 'current'
    elif v in ['previous', 1]:
        return 'previous'
    elif v in ['mean_over_interval', '_']:
        return v
    else:
        raise ValueError(
            f"Value of sampling method not supported, value is: {v}. Sampling method must be 'current' "
            f"(or 0 if sampling_method is int), 'previous' (or 1 if sampling_method is int) or 'mean_over_interval'. "
            f"In case of deactivated sampling (for outputs), sampling_method must be '_'.")


class Sampling:
    def __init__(self, unconstructed_inputs: list[str], unconstructed_outputs: list[str], time_step: Union[int, float],
                 ignore_nan: bool = False):
        """
        A class providing methods for sampling

        Args:
            unconstructed_inputs (list[str]): names of unconstructed (!) input features
            unconstructed_outputs (list[str]): names of unconstructed (!) output features
            time_step (Union[int, float]): sampling interval, multiple of sampling of data
            ignore_nan: If True, intermediate rows with NaN values will be dropped.
                        If False, an error is raised if NaNs are present in intermediate rows after processing.
                        Default is False.
        """
        self.inputs = unconstructed_inputs
        self.outputs = unconstructed_outputs
        self.time_step = time_step
        self.ignore_nan = ignore_nan

    def sample_df_according_to_timestep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        samples given data frame to the new grid defined by time_step

        Args:
            df: pandas DataFrame
        Returns:
            pd.DataFrame: DataFrame with the new sampling grid
        """
        filtering = (df.index - df.index[0]) % self.time_step == 0
        df = df[filtering]
        return df

    def previous_or_mean_in_sampling_methods(self) -> list[bool]:
        """
        checks if any input uses the sampling methods 'previous' or 'mean_over_interval'

        Returns:
             list[bool]: list of bool stating if the sampling method of an input is prev./mean (True) or not (False)
                         (list in the order of self.inputs)
        """
        # no import on module level possible due to circular import
        from physXAI.preprocessing.preprocessing import FeatureConstruction

        arr = []
        for fn in self.inputs:
            sm = FeatureConstruction.get_feature(fn).get_sampling_method()
            arr.append(sm in ['previous', 'mean_over_interval'])
        return arr

    def sample_unconstructed_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        extracts the unconstructed inputs from the given DataFrame, applies their corresponding sampling method and
        samples them to the target grid

        Args:
            df (pd.DataFrame): data
        Returns:
            pd.DataFrame: DataFrame (X) that solely contains all unconstructed inputs (with the correct sampling)
        """

        # no import on module level possible due to circular import
        from physXAI.preprocessing.preprocessing import FeatureConstruction

        # extract inputs from DataFrame and get target sampling grid
        X = df[self.inputs].copy()
        target_grid = self.sample_df_according_to_timestep(df).index

        # different inputs can have different sampling methods
        res = []
        features_without_constructed = [FeatureConstruction.get_feature(inp) for inp in self.inputs]
        for f in features_without_constructed:
            # only process inputs with sampling method mean_over_interval first since X cannot be sampled
            # to the actual required time steps until the intermediate values were taken into the mean
            if f.get_sampling_method() == 'mean_over_interval':
                res.append(get_mean_over_interval(X[[f.feature]], target_grid))

        # sample X to target grid
        X = self.sample_df_according_to_timestep(X)
        # process inputs with sampling methods 'current' and 'previous'
        for f in features_without_constructed:
            _x = X[[f.feature]]
            if f.get_sampling_method() == 'current':
                # no transformation needed
                res.append(_x)
            elif f.get_sampling_method() == 'previous':
                # shift by 1
                _x = _x.shift(1)
                _x = _x.iloc[1:]
                res.append(_x)
            elif f.get_sampling_method() == 'mean_over_interval':
                continue
            else:
                raise NotImplementedError(f"Sampling method '{f.get_sampling_method()}' not implemented.")
        # concatenate sampled input data
        X = pd.concat(res, axis=1)
        X = X.sort_index(ascending=True)

        # Sampling methods 'previous' and 'mean_over_interval' reduce available data points by 1.
        previous_or_mean = self.previous_or_mean_in_sampling_methods()
        if any(previous_or_mean):
            # if at least one of the features uses 'current' as sampling method, shorten X
            if not all(previous_or_mean):
                X = X.iloc[1:]

        # check for NaNs
        if X.isnull().values.any():
            if self.ignore_nan:
                X.dropna(inplace=True)
            else:
                raise ValueError(
                    "Data Error: The input data contains NaN values in intermediate rows. If this is intended, set "
                    "ignore_nan=True in PreprocessingSingleStep.")
        return X

    def sample_unconstructed_outputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        extracts the unconstructed outputs from the given DataFrame and samples them to the target grid

        Args:
            df (pd.DataFrame): data
        Returns:
            pd.DataFrame: DataFrame (y) that solely contains all unconstructed outputs
        """
        y = df[self.outputs].copy()
        y = self.sample_df_according_to_timestep(y)

        # Sampling methods 'previous' and 'mean_over_interval' reduce available data points by 1.
        # synchronize length of X and y
        if any(self.previous_or_mean_in_sampling_methods()):
            y = y.iloc[1:]

        # check for NaNs
        if y.isnull().values.any():
            if self.ignore_nan:
                y.dropna(inplace=True)
            else:
                raise ValueError(
                    "Data Error: The output data contains NaN values in intermediate rows. If this is intended,"
                    "set ignore_nan=True in PreprocessingSingleStep.")
        return y

    def sample_constructed_outputs(self, df: pd.DataFrame, constructed_outputs: list[str]) -> pd.DataFrame:
        """
        Correct shifting of constructed outputs if they are based on input features with sampling previous or mean_over_interval.
        Since the inputs are shifted before the constructed features are created, the constructed output has to be
        shifted to invert / neutralize the shift of the input features that was applied before.

        Args:
            df (pd.DataFrame): data including constructed features
            constructed_outputs (list[str]): names of constructed output features
        Returns:
            pd.DataFrame: modified DataFrame (df)
        """
        # no import on module level possible due to circular import
        from physXAI.preprocessing.preprocessing import FeatureConstruction, FeatureTwo

        if any(self.previous_or_mean_in_sampling_methods()):
            methods = ['previous', 'mean_over_interval']
            for out in constructed_outputs:
                out_feature = FeatureConstruction.get_feature(out)
                if isinstance(out_feature, FeatureTwo):
                    # correct shifting only if output bases on input features with before mentioned sampling methods
                    if (out_feature.feature1.get_sampling_method() in methods or
                            out_feature.feature2.get_sampling_method() in methods):
                        df[out_feature.feature] = df[out_feature.feature].shift(-1)
                else:  # constructed feature that doesn't consist of two features (FeatureExp, ...)
                    # correct shifting only if output bases on input features with before mentioned sampling methods
                    if out_feature.f1.get_sampling_method() in methods:
                        df[out_feature.feature] = df[out_feature.feature].shift(-1)
        return df


def get_mean_over_interval(x: pd.DataFrame, target_grid: pd.DataFrame.index) -> pd.DataFrame:
    """samples and returns x on target grid taking the mean over the interval (between the grid indices)"""

    def pairwise(iterable: Iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    original_grid = np.array(x.index)
    results = []
    for i, j in pairwise(target_grid):
        slicer = np.logical_and(original_grid >= i, original_grid < j)
        d = {'Index': j}
        for inp in x.columns:
            d[inp] = x[inp][slicer].mean()
        results.append(d)

    x = pd.DataFrame(results).set_index('Index')

    return x
