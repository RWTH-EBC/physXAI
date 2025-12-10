from abc import ABC, abstractmethod
from typing import Type, Union
import numpy as np
from pandas import DataFrame, Series


def _return_valid_sampling_method(v: Union[int, str]):
    """ check the validity of the given sampling method and return a string if val is int """

    if not isinstance(v, (int, str)):
        raise TypeError(f'Type of sampling method not supported. Type is {type(v)}, must be int or str.')

    if v in ['current', 0]:
        return 'current'
    elif v in ['previous', 1]:
        return 'previous'
    elif v == 'mean_over_interval':
        return 'mean_over_interval'
    else:
        raise ValueError(
            f"Value of sampling method not supported, value is: {v}. Sampling method must be 'current' "
            f"(or 0 if s is int), 'previous' (or 1 if s is int) or 'mean_over_interval'.")


class FeatureBase(ABC):
    """
    Abstract Base Class for all feature engineering components.
    Each feature object represents a column (or a transformation that results in a column)
    in a Pandas DataFrame. It supports arithmetic operations to combine features.
    """

    def __init__(self, name: str, sampling_method: Union[str, int] = None, **kwargs):
        """
        Initializes a FeatureBase instance.

        Args:
            name (str): The name of the feature. This will be the column name in the DataFrame.
            sampling_method (Union[str, int]): Time step of the input data used to predict the output.
                - if None: Feature._default_sampling_method is used
                - if 'current' or 0: Current time step will be used for prediction.
                - if 'previous' or 1: Previous time step will be used for prediction.
                - if 'mean_over_interval': Mean between current and previous time step will be used.
            **kwargs: Catches any additional keyword arguments.
        """

        self.feature: str = name
        self._sampling_method = None
        self.set_sampling_method(sampling_method)

        # Automatically registers the newly created feature instance with the FeatureConstruction manager
        FeatureConstruction.append(self)

    def get_sampling_method(self) -> str:
        """returns the Features sampling method"""
        return self._sampling_method

    def set_sampling_method(self, val: Union[str, int] = None):
        """
        Sets the feature's sampling method. If None is given, Feature._default_sampling_method is used
        Available methods:
        - 'current' or 0: Current time step will be used for prediction.
        - 'previous' or 1: Previous time step will be used for prediction.
        - 'mean_over_interval': Mean between current and previous time step will be used.
        """

        if val is None:
            self._sampling_method = Feature.get_default_sampling_method()
        else:
            self._sampling_method = _return_valid_sampling_method(val)

    def rename(self, name: str):
        """
        Renames the feature.

        Args:
            name (str): The new name for the feature.
        """

        self.feature = name

    def process(self, df: DataFrame) -> Series:
        """
         Processes the DataFrame to return the Series corresponding to this feature.
         For a base feature that already exists in the DataFrame, it simply returns the column.
         For derived features, this method would compute the feature if it doesn't exist.

         Args:
             df (DataFrame): The input DataFrame.

         Returns:
             Series: The Pandas Series representing this feature.
         """

        return df[self.feature]

    # --- Operator Overloading for Feature Arithmetic ---
    # These methods allow FeatureBase objects to be combined using standard arithmetic operators,
    # creating new composite feature objects (e.g., FeatureAdd, FeatureSub).
    def __add__(self, other):
        return FeatureAdd(self, other)

    def __radd__(self, other):
        return FeatureAdd(other, self)

    def __sub__(self, other):
        return FeatureSub(self, other)

    def __rsub__(self, other):
        return FeatureSub(other, self)

    def __mul__(self, other):
        return FeatureMul(self, other)

    def __rmul__(self, other):
        return FeatureMul(other, self)

    def __truediv__(self, other):
        return FeatureTrueDiv(self, other)

    def __rtruediv__(self, other):
        return FeatureTrueDiv(other, self)

    def __pow__(self, other):
        return FeaturePow(self, other)

    def exp(self):
        """Creates a new feature representing e^(self)."""
        return FeatureExp(self)

    def sin(self):
        """Creates a new feature representing sin(self)."""
        return FeatureSin(self)

    def cos(self):
        """Creates a new feature representing cos(self)."""
        return FeatureCos(self)

    def lag(self, lag: int, previous: bool = True):
        """
           Creates a lagged version of this feature.

           Args:
               lag (int): The number of time steps to lag by.
               previous (bool): If True and lag_value > 1, returns a list of FeatureLag objects
                                for all lags from 1 up to lag_value. Otherwise, returns a single
                                FeatureLag object for the specified lag_value.

           Returns:
               FeatureLag or List[FeatureLag]: A single lagged feature or a list of lagged features, each with the same
                                                sampling method as their corresponding base feature.
        """

        if previous and lag > 1:
            lg = list()
            for i in range(1, lag + 1):
                lg.append(FeatureLag(self, i))
            return lg
        else:
            return FeatureLag(self, lag)

    def get_config(self) -> dict:
        return {
            'class_name': self.__class__.__name__,
            'name': self.feature,
            'sampling_method': self.get_sampling_method(),
        }

    @classmethod
    def from_config(cls, config: dict) -> 'FeatureBase':
        return cls(**config)


# --- Registry for Feature Classes ---
# This registry maps class names (strings) to the actual class types (Type[FeatureBase]).
# It's used by `feature_from_config` to dynamically create instances of the correct feature class.
CONSTRUCTED_CLASS_REGISTRY: dict[str, Type['FeatureBase']] = dict()


def register_feature(cls):
    """
    A class decorator that registers the decorated class in the CONSTRUCTED_CLASS_REGISTRY.
    The class is registered using its __name__.
    """
    if cls.__name__ in CONSTRUCTED_CLASS_REGISTRY:  # pragma: no cover
        print(f"Warning: Class '{cls.__name__}' is already registered. Overwriting.")  # pragma: no cover
    CONSTRUCTED_CLASS_REGISTRY[cls.__name__] = cls
    return cls  # Decorators must return the class (or a replacement)


def feature_from_config(item_conf: dict) -> 'FeatureBase':
    """
    Factory function to create a feature object from its configuration dictionary.

    Args:
        item_conf (dict): The configuration dictionary for a single feature.
                          Must contain 'class_name' and other necessary parameters.

    Returns:
        FeatureBase: An instance of the appropriate feature subclass.

    Raises:
        KeyError: If 'class_name' is not in `item_conf` or if the class_name is not in `ITEM_CLASS_REGISTRY`.
    """
    class_name = item_conf['class_name']
    feature_class = CONSTRUCTED_CLASS_REGISTRY[class_name]
    f1f = feature_class.from_config(item_conf)
    return f1f


@register_feature
class Feature(FeatureBase):
    """
    Represents a basic feature that is assumed to exist directly in the input DataFrame.
    Its `process` method simply retrieves the column by its name.
    """

    _default_sampling_method = 'previous'

    @classmethod
    def get_default_sampling_method(cls):
        return Feature._default_sampling_method

    @classmethod
    def set_default_sampling_method(cls, val: Union[str, int]):
        """
        Sets the default sampling method for all features that do not have a custom sampling method. Available methods:
        - 'current' or 0: Current time step will be used for prediction.
        - 'previous' or 1: Previous time step will be used for prediction.
        - 'mean_over_interval': Mean between current and previous time step will be used.
        """
        Feature._default_sampling_method = _return_valid_sampling_method(val)


@register_feature
class FeatureLag(FeatureBase):
    """
    Represents a lagged version of another feature.
    Calculates `df[original_feature_name].shift(lag_steps)`.
    """

    def __init__(self, f: Union[FeatureBase, str], lag: int, name: str = None, **kwargs):
        """
        Initializes a FeatureLag instance.

        Args:
            f (FeatureBase or str): The original feature object or its name.
            lag (int): The number of time steps to lag by.
            name (str, optional): The name for this lagged feature. If None, it's auto-generated
                                  as "{original_name}_lag{X}".
            **kwargs: Catches any additional keyword arguments.
        """
        if isinstance(f, FeatureBase):
            self.origf: str = f.feature
            if name is None:
                name = f.feature + f'_lag{lag}'

            # lags must have the same sampling_method as their base feature
            sampling_method = f.get_sampling_method()
        else:
            self.origf: str = f
            if name is None:
                name = f + f'_lag{lag}'

            # lags must have the same sampling_method as their base feature
            sampling_method = FeatureConstruction.get_feature(f).get_sampling_method()

        if 'sampling_method' in kwargs.keys():
            assert kwargs['sampling_method'] == sampling_method, (
                f"lags must have the same sampling method as their base feature. Sampling method of base feature is"
                f" {sampling_method} but for lag {kwargs['sampling_method']} was given as sampling method."
            )
            kwargs.__delitem__('sampling_method')  # constructor must not get more than one arg with the same key

        super().__init__(name, sampling_method=sampling_method, **kwargs)
        self.lag: int = lag

    def process(self, df: DataFrame) -> Series:
        if self.feature not in df.columns:
            df[self.feature] = df[self.origf].shift(self.lag)
        return super().process(df)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'lag': self.lag, 'f': self.origf})
        return config


class FeatureTwo(FeatureBase, ABC):
    """
    Abstract Base Class for features derived from two other features (or constants).
    Examples: FeatureAdd (f1 + f2), FeatureSub (f1 - f2).
    """

    def __init__(self, feature1: Union[FeatureBase, int, float], feature2: Union[FeatureBase, int, float], name: str = None,
                 **kwargs):
        """
        Initializes a FeatureTwo instance.

        Args:
            feature1 (FeatureBase or int or float): The first operand.
            feature2 (FeatureBase or int or float): The second operand.
            name (str, optional): Name for the derived feature. If None, it's auto-generated
                                  by the `name` abstract method of the subclass.
            **kwargs: Catches any additional keyword arguments.
        """

        if isinstance(feature1, FeatureBase):
            f1n = feature1.feature
        else:
            f1n = str(feature1)
        if isinstance(feature2, FeatureBase):
            f2n = feature2.feature
        else:
            f2n = str(feature2)
        if name is None:
            name = self.name(f1n, f2n)
        super().__init__(name, **kwargs)
        self.feature1 = feature1
        self.feature2 = feature2

    def process(self, df: DataFrame) -> Series:
        """
        Calculates and returns the derived feature Series.
        If the column doesn't exist, it processes the operand features (if they are FeatureBase objects)
        or uses the constant values, then applies the `calc` method.

        Args:
            df (DataFrame): The input DataFrame.

        Returns:
            Series: The derived feature Series.
        """

        if self.feature not in df.columns:
            if isinstance(self.feature1, FeatureBase):
                f1 = self.feature1.process(df)
            else:
                f1 = self.feature1
            if isinstance(self.feature2, FeatureBase):
                f2 = self.feature2.process(df)
            else:
                f2 = self.feature2
            df[self.feature] = self.calc(f1, f2)
        return super().process(df)

    @abstractmethod
    def calc(self, f1, f2):
        """
        Abstract method to perform the actual calculation between the two processed operands.
        To be implemented by subclasses (e.g., addition, subtraction).

        Args:
            f1: The processed first operand (either a Series or a scalar).
            f2: The processed second operand (either a Series or a scalar).

        Returns:
            Series: The result of the calculation.
        """
        pass

    @abstractmethod
    def name(self, f1: str, f2: str) -> str:
        """
        Abstract method to generate a descriptive name for the derived feature,
        based on the names of its operands.

        Args:
            f1 (str): Name of the first operand.
            f2 (str): Name of the second operand.

        Returns:
            str: The auto-generated name for this feature.
        """
        pass

    def get_config(self) -> dict:
        """
        Returns the configuration for FeatureTwo.
        Includes configurations of its operand features if they are FeatureBase objects,
        or the constant values otherwise.
        """

        config = super().get_config()
        if isinstance(self.feature1, FeatureBase):
            f1n = self.feature1.feature
        else:
            f1n = self.feature1
        if isinstance(self.feature2, FeatureBase):
            f2n = self.feature2.feature
        else:
            f2n = self.feature2
        config.update({'feature1': f1n, 'feature2': f2n})
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'FeatureTwo':
        """
        Creates a FeatureTwo instance (or its subclass) from a configuration dictionary.
        Handles reconstruction of operand features if they were FeatureBase objects.

        Args:
            config (dict): Configuration dictionary. Must contain 'feature1' and 'feature2'.

        Returns:
            FeatureTwo: An instance of the specific FeatureTwo subclass.
        """

        # Reconstruct feature 1
        if isinstance(config['feature1'], dict):
            item_conf = config['feature1']
            # Check if feature already exists
            f1n = FeatureConstruction.get_feature(item_conf['name'])
            if f1n is None:
                f1n = feature_from_config(item_conf)
        elif isinstance(config['feature1'], str):
            f1n = FeatureConstruction.get_feature(config['feature1'])
        else:
            f1n = config['feature1']
        config['feature1'] = f1n

        # Reconstruct feature 2
        if isinstance(config['feature2'], dict):
            item_conf = config['feature2']
            # Check if feature already exists
            f2n = FeatureConstruction.get_feature(item_conf['name'])
            if f2n is None:
                f2n = feature_from_config(item_conf)
        elif isinstance(config['feature2'], str):
            f2n = FeatureConstruction.get_feature(config['feature2'])
        else:
            f2n = config['feature2']
        config['feature2'] = f2n

        return cls(**config)


@register_feature
class FeatureAdd(FeatureTwo):

    def calc(self, f1, f2):
        return f1 + f2

    def name(self, f1: str, f2: str) -> str:
        return '(' + f1 + '+' + f2 + ')'


@register_feature
class FeatureSub(FeatureTwo):

    def calc(self, f1, f2):
        return f1 - f2

    def name(self, f1: str, f2: str) -> str:
        return '(' + f1 + '-' + f2 + ')'


@register_feature
class FeatureMul(FeatureTwo):

    def calc(self, f1, f2):
        return f1 * f2

    def name(self, f1: str, f2: str) -> str:
        return '(' + f1 + '*' + f2 + ')'


@register_feature
class FeatureTrueDiv(FeatureTwo):

    def calc(self, f1, f2):
        return f1 / f2

    def name(self, f1: str, f2: str) -> str:
        return '(' + f1 + '/' + f2 + ')'


@register_feature
class FeaturePow(FeatureTwo):

    def calc(self, f1, f2):
        return f1 ** f2

    def name(self, f1: str, f2: str) -> str:
        return '(' + f1 + '**' + f2 + ')'


@register_feature
class FeatureExp(FeatureBase):
    """Feature representing e^(feature)."""

    def __init__(self, f1: FeatureBase, name: str = None, **kwargs):
        self.f1: FeatureBase = f1
        if name is None:
            name = 'exp(' + f1.feature + ')'
        super().__init__(name, **kwargs)

    def process(self, df: DataFrame) -> Series:
        if self.feature not in df.columns:
            df[self.feature] = np.exp(self.f1.process(df))
        return super().process(df)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'f1': self.f1.feature})
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'FeatureExp':
        item_conf = config['f1']
        f1n = FeatureConstruction.get_feature(item_conf)
        if f1n is None:
            f1n = feature_from_config(item_conf)
        config['f1'] = f1n
        return cls(**config)


@register_feature
class FeatureSin(FeatureBase):
    """Feature representing sin(feature)."""

    def __init__(self, f1: FeatureBase, name: str = None, **kwargs):
        self.f1: FeatureBase = f1
        if name is None:
            name = 'sin(' + f1.feature + ')'
        super().__init__(name, **kwargs)

    def process(self, df: DataFrame) -> Series:
        if self.feature not in df.columns:
            df[self.feature] = np.sin(self.f1.process(df))
        return super().process(df)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'f1': self.f1.feature})
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'FeatureSin':
        item_conf = config['f1']
        f1n = FeatureConstruction.get_feature(item_conf)
        if f1n is None:
            f1n = feature_from_config(item_conf)
        config['f1'] = f1n
        return cls(**config)


@register_feature
class FeatureCos(FeatureBase):
    """Feature representing cos(feature)."""

    def __init__(self, f1: FeatureBase, name: str = None, **kwargs):
        self.f1: FeatureBase = f1
        if name is None:
            name = 'cos(' + f1.feature + ')'
        super().__init__(name, **kwargs)

    def process(self, df: DataFrame) -> Series:
        if self.feature not in df.columns:
            df[self.feature] = np.cos(self.f1.process(df))
        return super().process(df)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'f1': self.f1.feature})
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'FeatureCos':
        item_conf = config['f1']
        f1n = FeatureConstruction.get_feature(item_conf)
        if f1n is None:
            f1n = feature_from_config(item_conf)
        config['f1'] = f1n
        return cls(**config)


@register_feature
class FeatureConstant(FeatureBase):
    """
    Represents a feature that is a constant value across all rows.
    """

    def __init__(self, c: float, name: str, **kwargs):
        self.c = c
        super().__init__(name, **kwargs)

    def process(self, df: DataFrame) -> Series:
        if self.feature not in df.columns:
            df[self.feature] = self.c
        return super().process(df)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'c': self.c})
        return config


class FeatureConstruction:
    """
    Manages a collection of feature engineering objects (subclasses of FeatureBase).
    Provides methods to process a DataFrame to generate all registered features,
    and to save/load the feature engineering pipeline configuration.
    """

    features = list[FeatureBase]()
    _default_sampling_method = 'previous'

    @staticmethod
    def set_default_sampling_method(val: Union[str, int]):
        """
        Sets the default sampling method for all features that do not have a custom sampling method. Available methods:
        - 'current' or 0: Current time step will be used for prediction.
        - 'previous' or 1: Previous time step will be used for prediction.
        - 'mean_over_interval': Mean between current and previous time step will be used.
        """
        FeatureConstruction._default_sampling_method = _return_valid_sampling_method(val)

    @staticmethod
    def reset():
        """Clears all registered features and input names. Furthermore, resets the default sampling method"""
        FeatureConstruction.features = list[FeatureBase]()
        Feature.set_default_sampling_method('previous')

    @staticmethod
    def append(f: FeatureBase):
        """
        Adds a feature object to the list of managed features.
        Called automatically from FeatureBase.__init__.

        Args:
            f (FeatureBase): The feature object to add.
        """
        if FeatureConstruction.get_feature(f.feature) is None:
            FeatureConstruction.features.append(f)

    @staticmethod
    def get_feature(name: str) -> Union[FeatureBase, None]:
        """
        Retrieves a feature object by its name from the managed list.

        Args:
            name (str): The name of the feature to retrieve.

        Returns:
            FeatureBase or None: The found feature object, or None if not found.
        """
        for f in FeatureConstruction.features:
            if f.feature == name:
                return f
        return None

    @staticmethod
    def get_features_including_lagged_features(l: list[str] = None) -> list[str]:
        """
        returns a list of the names of all FeatureLag and FeatureTwo where at least one feature is a FeatureLag
        - within the given list or
        - of all constructed features if list is None

        Args:
            l (list[str]): list of feature names to search in

        Returns:
            list[str]: the list of lag-based features
        """

        # if no list is given, search in all features
        if not l:
            l = FeatureConstruction.features

        def recursive_search(feature):
            """Recursively checks for lagged features"""
            if isinstance(feature, FeatureLag):
                return True

            elif isinstance(feature, FeatureTwo):
                # Check both sub-features recursively
                return recursive_search(feature.feature1) or recursive_search(feature.feature2)

            return False

        res = list()
        for f in FeatureConstruction.features:
            if isinstance(f, FeatureLag) and (f.feature in l):
                res.append(f.feature)  # name of the feature

            elif isinstance(f, FeatureTwo) and (f.feature in l):
                # Use recursive search to check for nested lagged features
                if recursive_search(f.feature1) or recursive_search(f.feature2):
                    res.append(f.feature)

        return res

    @staticmethod
    def process_inputs(inputs: list[Union[str, FeatureBase]]) -> list[str]:
        """
        Creates a Feature for all inputs that are not yet created as features

        Args:
             inputs (list(Union[str, FeatureBase])): List of column names or Features to be used as input features.

        Returns:
            list[str]: list of column names of all input features
        """

        input_str = list()

        for inp in inputs:
            if isinstance(inp, FeatureBase):
                input_str.append(inp.feature)  # get name of feature (which is used as column name)
            elif isinstance(inp, str):
                input_str.append(inp)
                # check if a Feature with the given name (inp) was already created, otherwise create it
                if not any(inp == f.feature for f in FeatureConstruction.features):
                    Feature(name=inp)
            else:
                raise TypeError(f"Only inputs with types 'str' or 'FeatureBase' allowed, got type {type(inp)} instead")

        return input_str

    @staticmethod
    def process(df: DataFrame, feature_names: list[str] = None):
        """
        Processes the input DataFrame by applying all registered feature transformations in order.
        Each feature's `process` method is called, which typically adds a new column to `df`
        if it doesn't already exist.

        Args:
            df (DataFrame): The DataFrame to process and add features to.
            feature_names (list[str]): optional parameter to only process those features given in feature_names
        """

        if feature_names is None:
            for f in FeatureConstruction.features:
                f.process(df)
        else:
            for f in FeatureConstruction.features:
                if f.feature in feature_names:
                    f.process(df)

    @staticmethod
    def get_config() -> list:
        """
        Returns a list of configuration dictionaries for all managed features.
        This list can be serialized (e.g., to JSON) to save the feature pipeline.
        """

        item_configs = [item.get_config() for item in FeatureConstruction.features]
        return item_configs

    @staticmethod
    def from_config(config: list):
        """
        Reconstructs the feature engineering pipeline from a list of configuration dictionaries.
        Clears any existing features and populates `FeatureConstruction.features` with
        newly created feature objects based on the provided configurations.

        Args:
            config (List[dict]): A list where each dictionary is the configuration
                                      for a single feature object.
        """

        FeatureConstruction.features = list[FeatureBase]()
        for item_conf in config:
            f = FeatureConstruction.get_feature(item_conf['name'])
            if f is None:
                feature_from_config(item_conf)
