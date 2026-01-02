import functools
from itertools import combinations
from abc import ABC, abstractmethod
import operator
import os
from pathlib import Path
from typing import Optional, Union
from copy import deepcopy

import numpy as np
from physXAI.models.ann.keras_models.keras_models import NonNegPartial
from physXAI.models.modular.modular_expression import (ModularExpression, register_modular_expression,
                                                       get_modular_expressions_by_name)
from physXAI.models.ann.ann_design import ANNModel, CMNNModel, ClassicalANNModel
from physXAI.models.models import AbstractModel, LinearRegressionModel, register_model
from physXAI.preprocessing.training_data import TrainingDataGeneric
from physXAI.preprocessing.constructed import FeatureBase
from physXAI.utils.logging import Logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras import Sequential
from keras.src import Functional
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


@register_model
class ModularANN(ANNModel):
    """
    A Modular Artificial Neural Network model that allows for custom architectures.
    """

    def __init__(self, architecture: ModularExpression, batch_size: int = 32, epochs: int = 1000,
                 learning_rate: float = 0.001, early_stopping_epochs: Optional[int] = 100,
                 random_seed: int = 42, rescale_output: bool = False, **kwargs):
        """
        Initializes the ModularANN.

        Args:
            architecture (ModularExpression): The modular architecture defining the model.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of times to iterate over the entire training dataset.
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_epochs (int): Number of epochs with no improvement after which training will be stopped.
                                         If None, early stopping is disabled.
            random_seed (int): Seed for random number generators to ensure reproducibility.
            rescale_output (bool): Whether to rescale the output to output scale.
        """

        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed)
        self.architecture: ModularExpression = architecture

        self.rescale_output = rescale_output

        self.model_config.update({
            'rescale_output': rescale_output,  # the rest of the parameters are passed on to super
        })

    def generate_model(self, **kwargs):
        """
        Generates the Keras model using the specified modular architecture.
        """

        td = kwargs['td']
        n_features = td.X_train_single.shape[1]
        input_layer = keras.layers.Input(shape=(n_features,))
        x = self.architecture.construct(input_layer, td)
        if self.rescale_output:
            rescale_mean = float(np.mean(td.y_train_single))
            rescale_sigma = float(np.std(td.y_train_single, ddof=1))
            x = keras.layers.Rescaling(scale=rescale_sigma, offset=rescale_mean)(x)
        model = keras.models.Model(inputs=input_layer, outputs=x)
        model.summary()

        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'architecture': self.architecture.name,
            'rescale_output': self.rescale_output,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'ModularANN':

        a = ModularExpression.get_existing_modular_expression(config['architecture'])
        assert a is not None, (f"ModularExpression {config['architecture']} not found, make sure to construct required "
                               f"modular expressions before constructing {cls.__name__}.")
        config['architecture'] = a

        return cls(**config)


class ModularAbstractModel(ModularExpression, ABC):
    """
    Abstract Base Class for ModularExpressions having other ModularExpressions as inputs
    Examples: ModularModel, ModularExistingModel, ModularLinear, ...
    """
    def __init__(self, inputs: list[Union[ModularExpression, FeatureBase]], name: str):
        super().__init__(name)
        self.inputs = [inp if isinstance(inp, ModularExpression) else inp.input() for inp in inputs]

    @abstractmethod
    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        pass

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'inputs': [inp.name for inp in self.inputs],
        })
        return c

    @classmethod
    def _from_config(cls, item_config: dict, config: list[dict]) -> 'ModularAbstractModel':
        """
        Creates a ModularAbstractModel instance (or its subclass) from a configuration dictionary.
        Handles reconstruction of inputs.

        Args:
            item_config (dict): Configuration dictionary. Must contain key 'inputs' with list of input names
            config (list[dict]): The list with the configuration dictionaries of all modular expressions

        Returns:
            ModularAbstractModel: An instance of the specific ModularAbstractModel subclass.
        """

        item_config['inputs'] = get_modular_expressions_by_name(item_config['inputs'], config)
        return cls(**item_config)


@register_modular_expression
class ModularModel(ModularAbstractModel):

    allowed_models = [ClassicalANNModel, CMNNModel, LinearRegressionModel]
    i = 0

    def __init__(self, model: ANNModel, inputs: list[ModularExpression, FeatureBase], name: str = None,
                 nominal_range: tuple[float, float] = None):
        if not any(isinstance(model, allowed) for allowed in self.allowed_models):
            raise NotImplementedError(f"Currently {type(model)} is not supported. Allowed models are: {self.allowed_models}")

        if name is None:
            name = f"ModularModel_{ModularModel.i}"
            ModularModel.i += 1
        super().__init__(inputs, name)

        self.model = model
        self.model.model_config.update({
            "normalize": False,
            "rescale_output": False
        })
        self._nominal_range = nominal_range

        if nominal_range is None:
            self.rescale_output = False
        elif nominal_range is not None and len(nominal_range) != 2:
            raise ValueError(f"Modular Model: nominal_range must be a tuple of (min, max), but was {nominal_range}")
        else:
            self.rescale_output = True
            self.nominal_mean = (nominal_range[1] + nominal_range[0]) / 2.0
            self.nominal_sigma = (nominal_range[1] - nominal_range[0]) / 4.0  # Assuming 4 sigma covers the range

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)
            self.model.model_config['n_features'] = len(inps)
            td = deepcopy(td)
            td.columns = [inp.name for inp in self.inputs]
            if isinstance(self.model, LinearRegressionModel):
                lr = ModularLinear(inputs=self.inputs, name=self.name + "_linear").construct(input_layer, td)
                l = lr(keras.layers.Concatenate()(inps))
            else:
                l = self.model.generate_model(td=td)(keras.layers.Concatenate()(inps))
            if self.rescale_output:
                l = keras.layers.Rescaling(scale=self.nominal_sigma, offset=self.nominal_mean)(l) 
            ModularExpression.models[self.name] = l
            return l

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'model': self.model.get_config(),
            'nominal_range': self._nominal_range,
        })
        return c

    @classmethod
    def _from_config(cls, item_config: dict, config: list[dict]) -> 'ModularModel':
        """
        Creates a ModularModel instance from a configuration dictionary.
        Handles reconstruction of model (ANNModel) and inputs.

        Args:
            item_config (dict): Configuration dictionary. Must contain configuration for model as well.
            config (list[dict]): The list with the configuration dictionaries of all modular expressions

        Returns:
            ModularModel: An instance of the specific ModularModel.
        """

        assert isinstance(item_config['model'], dict), (f"config must contain the configuration (dict) for the model "
                                                        f"but config['model'] is {item_config['model']}]")
        m = AbstractModel.model_from_config(item_config['model'])
        item_config['model'] = m

        item_config['inputs'] = get_modular_expressions_by_name(item_config['inputs'], config)

        return cls(**item_config)


@register_modular_expression
class ModularExistingModel(ModularAbstractModel):

    def __init__(self, model: Union[Sequential, Functional, str, Path],
                 original_inputs: list[ModularExpression, FeatureBase], trainable: bool, name: str = None):
        if isinstance(model, str) or isinstance(model, Path):
            self.model_path = model
            model = keras.models.load_model(model)
        self.model = model

        if name is None:
            name = model.name + '_existing'
        super().__init__(original_inputs, name)

        self.model.trainable = trainable
        if not trainable:
            for layer in self.model.layers:
                layer.trainable = False

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)
            l = self.model(keras.layers.Concatenate()(inps))
            ModularExpression.models[self.name] = l
            return l

    def _get_config(self) -> dict:
        c = super()._get_config()

        # if model wasn't loaded from path originally, save it and store path
        if not hasattr(self, 'model_path'):
            self.model_path = Logger.get_model_savepath(save_name_model=self.model.name)
            self.model.save(self.model_path)

        c.update({
            'model': self.model_path,
            'original_inputs': c['inputs'],
            'trainable': self.model.trainable
        })
        c.__delitem__('inputs')  # super config contains key 'inputs', here key must be original_inputs
        return c

    @classmethod
    def _from_config(cls, item_config: dict, config: list[dict]) -> 'ModularExistingModel':
        """
        Creates a ModularExistingModel instance from a configuration dictionary.
        Handles reconstruction of original_inputs.

        Args:
            item_config (dict): Configuration dictionary
            config (list[dict]): The list with the configuration dictionaries of all modular expressions

        Returns:
            ModularExistingModel: An instance of the specific ModularExistingModel.
        """

        item_config['original_inputs'] = get_modular_expressions_by_name(item_config['original_inputs'], config)

        return cls(**item_config)


@register_modular_expression
class ModularLinear(ModularAbstractModel):
    i = 0

    def __init__(self, inputs: list[ModularExpression, FeatureBase], name: str = None,
                 nominal_range: tuple[float, float] = None):
        if name is None:
            name = f"ModularLinear_{ModularLinear.i}"
            ModularLinear.i += 1
        super().__init__(inputs, name)
        self._nominal_range = nominal_range

        if nominal_range is None:
            self.rescale_output = False
        elif nominal_range is not None and len(nominal_range) != 2:
            raise ValueError(f"Modular Model: nominal_range must be a tuple of (min, max), but was {nominal_range}")
        else:
            self.rescale_output = True
            self.nominal_mean = (nominal_range[1] + nominal_range[0]) / 2.0
            self.nominal_sigma = (nominal_range[1] - nominal_range[0]) / 4.0  # Assuming 4 sigma covers the range
        
    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)
            l = keras.layers.Dense(units=1, activation='linear')(keras.layers.Concatenate()(inps))
            if self.rescale_output:
                l = keras.layers.Rescaling(scale=self.nominal_sigma, offset=self.nominal_mean)(l) 
            ModularExpression.models[self.name] = l
            return l

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'nominal_range': self._nominal_range,
        })
        return c


@register_modular_expression
class ModularMonotoneLinear(ModularAbstractModel):
    i = 0

    def __init__(self, inputs: list[Union[ModularExpression, FeatureBase]], name: str = None,
                 monotonicities: Optional[dict[str, int]] = None, nominal_range: tuple[float, float] = None):
        if name is None:
            name = f"ModularMonotoneLinear_{ModularLinear.i}"
            ModularLinear.i += 1
        super().__init__(inputs, name)
        self._nominal_range = nominal_range

        if monotonicities is None:
            monotonicities = [0] * len(self.inputs)
        else:
            monotonicities = [0 if inp.name not in monotonicities.keys() else monotonicities[inp.name] for inp in self.inputs]
        self.monotonicities = monotonicities

        if nominal_range is None:
            self.rescale_output = False
        elif nominal_range is not None and len(nominal_range) != 2:
            raise ValueError(f"Modular Model: nominal_range must be a tuple of (min, max), but was {nominal_range}")
        else:
            self.rescale_output = True
            self.nominal_mean = (nominal_range[1] + nominal_range[0]) / 2.0
            self.nominal_sigma = (nominal_range[1] - nominal_range[0]) / 4.0  # Assuming 4 sigma covers the range
        
    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)
            l = keras.layers.Dense(units=1, activation='linear', kernel_constraint=NonNegPartial(self.monotonicities))(keras.layers.Concatenate()(inps))
            if self.rescale_output:
                l = keras.layers.Rescaling(scale=self.nominal_sigma, offset=self.nominal_mean)(l) 
            ModularExpression.models[self.name] = l
            return l

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'nominal_range': self._nominal_range,
            'monotonicities': self.monotonicities,
        })
        return c


@register_modular_expression
class ModularPolynomial(ModularAbstractModel):
    i = 0

    def __init__(self, inputs: list[ModularExpression, FeatureBase], degree: int = 2, interaction_degree: int = 1,
                 name: str = None, nominal_range: tuple[float, float] = None):
        if name is None:
            name = f"ModularPolynomial_{ModularPolynomial.i}"
            ModularPolynomial.i += 1
        super().__init__(inputs, name)
        assert degree >= 1, "Degree must be at least 1."
        assert interaction_degree >= 1, "Interaction degree must be at least 1."
        self.degree = degree
        self.interaction_degree = interaction_degree
        self._nominal_range = nominal_range

        if nominal_range is None:
            self.rescale_output = False
        elif nominal_range is not None and len(nominal_range) != 2:
            raise ValueError(f"Modular Model: nominal_range must be a tuple of (min, max), but was {nominal_range}")
        else:
            self.rescale_output = True
            self.nominal_mean = (nominal_range[1] + nominal_range[0]) / 2.0
            self.nominal_sigma = (nominal_range[1] - nominal_range[0]) / 4.0  # Assuming 4 sigma covers the range

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)

            new_features = list(inps)
            for feature in inps:
                for d in range(2, self.degree + 1):
                    new_features.append(feature ** d)
            for k in range(2, self.interaction_degree + 1):
                for combo in combinations(inps, k):
                    interaction_term = functools.reduce(operator.mul, combo)
                    new_features.append(interaction_term)

            l = keras.layers.Dense(units=1, activation='linear')(keras.layers.Concatenate()(new_features))
            if self.rescale_output:
                l = keras.layers.Rescaling(scale=self.nominal_sigma, offset=self.nominal_mean)(l) 
            ModularExpression.models[self.name] = l
            return l

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'degree': self.degree,
            'interaction:degree': self.interaction_degree,
            'nominal_range': self._nominal_range,
        })
        return c


@register_modular_expression
class ModularAverage(ModularAbstractModel):
    i = 0

    def __init__(self, inputs: list[ModularExpression, FeatureBase], name: str = None):
        if name is None:
            name = f"ModularAverage_{ModularAverage.i}"
            ModularAverage.i += 1
        super().__init__(inputs, name)

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)
            l = keras.layers.Average()(inps)
            ModularExpression.models[self.name] = l
            return l


@register_modular_expression
class ModularNormalization(ModularAbstractModel):
    i = 0

    def __init__(self, input: ModularExpression, name: str = None):
        if name is None:
            name = f"ModularNormalization_{ModularNormalization.i}"
            ModularNormalization.i += 1
        super().__init__([input], name)

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        inp = self.inputs[0].construct(input_layer, td)
        normalization = keras.layers.BatchNormalization()
        l = normalization(inp)
        return l

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'input': c['inputs'][0],
        })
        c.__delitem__('inputs')  # super config contains key 'inputs', here only single input
        return c

    @classmethod
    def _from_config(cls, item_config: dict, config: list[dict]) -> 'ModularNormalization':
        """
        Creates a ModularNormalization instance from a configuration dictionary.
        Handles reconstruction of single input.

        Args:
            item_config (dict): Configuration dictionary
            config (list[dict]): The list with the configuration dictionaries of all modular expressions

        Returns:
            ModularNormalization: An instance of the specific ModularNormalization.
        """

        item_config['input'] = get_modular_expressions_by_name(item_config['input'], config)[0]

        return cls(**item_config)
