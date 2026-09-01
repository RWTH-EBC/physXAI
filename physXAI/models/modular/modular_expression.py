from abc import ABC, abstractmethod
import os
from typing import Union, Type
import numpy as np
from physXAI.models.ann.keras_models.keras_models import ConstantLayer, DivideLayer, InputSliceLayer, PowerLayer
from physXAI.preprocessing.training_data import TrainingDataGeneric
import casadi as ca
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class ModularExpression(ABC):

    feature_list = dict()
    feature_list_normalized = dict()
    trainable_parameters = dict()
    models = dict()
    modular_expression_list = list['ModularExpression']()

    def __init__(self, name: str):
        self.name = name
        ModularExpression.modular_expression_list.append(self)

    @staticmethod
    def reset():
        ModularExpression.feature_list = dict()
        ModularExpression.feature_list_normalized = dict()
        ModularExpression.trainable_parameters = dict()
        ModularExpression.models = dict()
        ModularExpression.modular_expression_list = list()

    @abstractmethod
    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        pass
    
    def __add__(self, other):
        return ModularAdd(self, other)

    def __radd__(self, other):
        return ModularAdd(other, self)

    def __sub__(self, other):
        return ModularSub(self, other)

    def __rsub__(self, other):
        return ModularSub(other, self)

    def __mul__(self, other):
        return ModularMul(self, other)

    def __rmul__(self, other):
        return ModularMul(other, self)

    def __truediv__(self, other):
        return ModularTrueDiv(self, other)

    def __rtruediv__(self, other):
        return ModularTrueDiv(other, self)

    def __pow__(self, other):
        return ModularPow(self, other)
    
    def __rpow__(self, other):
        return ModularPow(other, self)
    
    def rename(self, name: str):
        self.name = name

    def _get_config(self) -> dict:
        c = {
            'class_name': self.__class__.__name__,
            'name': self.name,
        }
        return c

    @classmethod
    def _from_config(cls, item_config: dict, config: list[dict]) -> 'ModularExpression':
        return cls(**item_config)

    @staticmethod
    def get_config() -> list:
        """
        Returns a list of configuration dictionaries for all managed modular expressions.
        This list can be serialized (e.g., to JSON) to save the modular expression pipeline.
        """

        item_configs = [item._get_config() for item in ModularExpression.modular_expression_list]
        return item_configs

    @staticmethod
    def from_config(config: list):
        """
        Reconstructs the modular expression engineering pipeline from a list of configuration dictionaries.
        Clears any existing modular expressions and populates `ModularExpression.modular_expression_list` with
        newly created modular expression objects based on the provided configurations.

        Args:
            config (List[dict]): A list where each dictionary is the configuration
                                    for a single modular expression object.
        """

        ModularExpression.reset()
        for item_conf in config:
            f = ModularExpression.get_existing_modular_expression(item_conf['name'])
            if f is None:
                modular_expression_from_config(item_conf, config)

    @staticmethod
    def get_existing_modular_expression(name: str) -> Union['ModularExpression', None]:
        """
        Retrieves a modular expression object by its name from the managed list.

        Args:
            name (str): The name of the modular expression to retrieve.

        Returns:
            ModularExpression or None: The found modular expression object, or None if not found.
        """
        for f in ModularExpression.modular_expression_list:
            if f.name == name:
                return f
        return None
    
    def get_value(self, td: TrainingDataGeneric, input_layer: keras.layers.Input, sym_raw: dict, X_raw: dict):
        raise NotImplementedError("get_value method is only implemented for base expressions.")


def get_name(feature: Union[ModularExpression, int, float]) -> str:
    if isinstance(feature, ModularExpression):
        return feature.name
    else:
        return str(feature)


# --- Registry for ModularExpression Classes ---
# This registry maps class names (strings) to the actual class types (Type[ModularExpression]).
# It's used by `modular_expression_from_config` to dynamically create instances of the correct modular expression class.
CONSTRUCTED_CLASS_REGISTRY: dict[str, Type['ModularExpression']] = dict()


def modular_expression_from_config(item_conf: dict, config: list[dict]) -> 'ModularExpression':
    """
    Factory function to create a modular expression object from its configuration dictionary.

    Args:
        item_conf (dict): The configuration dictionary for a single modular expression.
                          Must contain 'class_name' and other necessary parameters.
        config (list[dict]): The list with the configuration dictionaries of all modular expressions

    Returns:
        ModularExpression: An instance of the appropriate modular expression subclass.

    Raises:
        KeyError: If 'class_name' is not in `item_conf` or if the class_name is not in `CONSTRUCTED_CLASS_REGISTRY`.
    """
    class_name = item_conf['class_name']
    modular_expression_class = CONSTRUCTED_CLASS_REGISTRY[class_name]
    item_conf.__delitem__('class_name')
    f1f = modular_expression_class._from_config(item_conf, config)
    return f1f


def get_modular_expressions_by_name(names: Union[str, list[str]], config: list[dict]) -> list[ModularExpression]:
    """
    Retrieves modular expressions by their names if they have already been constructed,
    otherwise constructs the modular expression objects based on the given configuration.

    Args:
        names (Union[str, list[str]]): single name (str) or list of names of the modular expressions to retrieve
        config (list[dict]): The list with the configuration dictionaries of all modular expressions

    Returns:
        ModularExpression: An instance of the specific ModularExpression subclass.
    """

    if isinstance(names, str):  # convert str to list
        names = [names]

    l = list[ModularExpression]()
    for name in names:
        me = ModularExpression.get_existing_modular_expression(
            name)  # if modular expression already constructed, retrieve it

        if me is None:  # modular expression yet unconstructed
            item_config = dict()
            for item in config:  # find config of modular expression to construct it
                if item['name'] == name:
                    item_config = item
            me = modular_expression_from_config(item_config, config)  # construct modular expression
        l.append(me)
    return l


def register_modular_expression(cls):
    """
    A class decorator that registers the decorated class in the CONSTRUCTED_CLASS_REGISTRY.
    The class is registered using its __name__.
    """
    if cls.__name__ in CONSTRUCTED_CLASS_REGISTRY:  # pragma: no cover
        print(f"Warning: Class '{cls.__name__}' is already registered. Overwriting.")  # pragma: no cover
    CONSTRUCTED_CLASS_REGISTRY[cls.__name__] = cls
    return cls  # Decorators must return the class (or a replacement)


@register_modular_expression
class ModularFeature(ModularExpression):

    def __init__(self, name: str, normalize: bool = True):
        super().__init__(name)
        self.normalize = normalize

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.normalize and self.name in ModularExpression.feature_list_normalized.keys():
            return ModularExpression.feature_list_normalized[self.name]
        elif not self.normalize and self.name in ModularExpression.feature_list.keys():
            return ModularExpression.feature_list[self.name]
        else:
            x = InputSliceLayer([td.columns.index(self.name)])(input_layer)
            if self.normalize:
                l = keras.layers.Normalization()
                l.adapt(td.X_train_single[:, td.columns.index(self.name)].reshape(-1, 1))
                x = l(x)
                ModularExpression.feature_list_normalized[self.name] = x
            else:
                ModularExpression.feature_list[self.name] = x

            return x
        
    def get_value(self, td: TrainingDataGeneric, input_layer: keras.layers.Input, sym_raw: dict, X_raw: dict):
        if not self.normalize and self.name in ModularExpression.feature_list.keys():
            model = ModularExpression.feature_list[self.name]
        elif self.normalize and self.name in ModularExpression.feature_list_normalized.keys():
            model = ModularExpression.feature_list_normalized[self.name]
        else:
            raise ValueError(f"Feature '{self.name}' not found in feature lists. Make sure to construct the modular expression pipeline before trying to get feature values.")
        
        model = keras.models.Model(inputs=input_layer, outputs=model)

        X = model.predict(td.X_train_single, verbose=0)

        if self.name not in sym_raw.keys():
            sym_raw[self.name] = ca.MX.sym(self.name)
            X_raw[self.name] = X
        sym = sym_raw[self.name]

        return X, sym

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'normalize': self.normalize,
        })
        return c


@register_modular_expression
class ModularTrainable(ModularExpression):

    i = 0

    def __init__(self, name: str = None, initial_value: float = None, trainable: bool = True):
        if name is None:
            name = f"ModularTrainable_{ModularTrainable.i}"
            ModularTrainable.i += 1
        super().__init__(name)
        self.initial_value = initial_value
        self.trainable = trainable

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.trainable_parameters.keys():
            return ModularExpression.trainable_parameters[self.name]
        else:
            l = ConstantLayer(trainable=self.trainable, weight_name=self.name, value=self.initial_value)(input_layer)
            ModularExpression.trainable_parameters[self.name] = l
            return l

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'initial_value': self.initial_value,
            'trainable': self.trainable,
        })
        return c


class ModularTwo(ModularExpression, ABC):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str):
        super().__init__(name)
        self.feature1 = feature1
        self.feature2 = feature2

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if isinstance(self.feature1, (int, float)):
            l1 = ConstantLayer(value=self.feature1)(input_layer)
        else:
            l1 = self.feature1.construct(input_layer, td)

        if isinstance(self.feature2, (int, float)):
            l2 = ConstantLayer(value=self.feature2)(input_layer)
        else:
            l2 = self.feature2.construct(input_layer, td)

        return self._construct(l1, l2)
    
    @abstractmethod
    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        pass

    def get_value(self, td: TrainingDataGeneric, input_layer: keras.layers.Input, sym_raw: dict, X_raw: dict):
        if isinstance(self.feature1, (int, float)):
            val1 = self.feature1
            sym1 = self.feature1
        else:
            val1, sym1 = self.feature1.get_value(td, input_layer, sym_raw, X_raw)

        if isinstance(self.feature2, (int, float)):
            val2 = self.feature2
            sym2 = self.feature2
        else:
            val2, sym2 = self.feature2.get_value(td, input_layer, sym_raw, X_raw)

        return self._get_value(val1, val2), self._get_value(sym1, sym2)
    
    @abstractmethod
    def _get_value(self, val1, val2):
        pass

    def _get_config(self) -> dict:
        c = super()._get_config()
        if isinstance(self.feature1, ModularExpression):
            f1n = self.feature1.name
        else:
            f1n = self.feature1
        if isinstance(self.feature2, ModularExpression):
            f2n = self.feature2.name
        else:
            f2n = self.feature2
        c.update({
            'feature1': f1n,
            'feature2': f2n,
        })
        return c

    @classmethod
    def _from_config(cls, item_config: dict, config: list[dict]) -> 'ModularTwo':
        """
        Creates a ModularTwo instance (or its subclass) from a configuration dictionary.
        Handles reconstruction of operand modular expressions if they were ModularExpression objects.

        Args:
            item_config (dict): Configuration dictionary. Must contain 'feature1' and 'feature2'.
            config (list[dict]): The list with the configuration dictionaries of all modular expressions

        Returns:
            ModularTwo: An instance of the specific ModularTwo subclass.
        """

        # Reconstruct feature 1
        if isinstance(item_config['feature1'], dict):
            feature_conf = item_config['feature1']
            # Check if modular expression already exists
            f1n = ModularExpression.get_existing_modular_expression(feature_conf['name'])
            if f1n is None:
                f1n = modular_expression_from_config(feature_conf, config)
        elif isinstance(item_config['feature1'], str):
            f1n = ModularExpression.get_existing_modular_expression(item_config['feature1'])
        else:  # feature is int or float
            f1n = item_config['feature1']
        item_config['feature1'] = f1n

        # Reconstruct feature 2
        if isinstance(item_config['feature2'], dict):
            feature_conf = item_config['feature2']
            # Check if modular expression already exists
            f2n = ModularExpression.get_existing_modular_expression(feature_conf['name'])
            if f2n is None:
                f2n = modular_expression_from_config(feature_conf, config)
        elif isinstance(item_config['feature2'], str):
            f2n = ModularExpression.get_existing_modular_expression(item_config['feature2'])
        else:  # feature is int or float
            f2n = item_config['feature2']
        item_config['feature2'] = f2n

        return cls(**item_config)


@register_modular_expression
class ModularAdd(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}+{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return keras.layers.Add()([layer1, layer2])
    
    def _get_value(self, val1, val2):
        return val1 + val2
    

@register_modular_expression
class ModularSub(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}-{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return keras.layers.Subtract()([layer1, layer2])
    
    def _get_value(self, val1, val2):
        return val1 - val2
    

@register_modular_expression
class ModularMul(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}*{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return keras.layers.Multiply()([layer1, layer2])
    
    def _get_value(self, val1, val2):
        return val1 * val2
    

@register_modular_expression
class ModularTrueDiv(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}/{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return DivideLayer()([layer1, layer2])
    
    def _get_value(self, val1, val2):
        return val1 / val2
    

@register_modular_expression
class ModularPow(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}**{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return PowerLayer()([layer1, layer2])

    def _get_value(self, val1, val2):
        return val1 ** val2


class ModularOne(ModularExpression, ABC):
    """
    Abstract Base Class for ModularExpressions applying a single argument function to one operand.
    Examples: ModularActivation
    """

    def __init__(self, feature: Union[ModularExpression, int, float], name: str):
        super().__init__(name)
        self.feature = feature

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if isinstance(self.feature, (int, float)):
            l = ConstantLayer(value=self.feature)(input_layer)
        else:
            l = self.feature.construct(input_layer, td)

        return self._construct(l)

    @abstractmethod
    def _construct(self, layer: keras.layers.Layer) -> keras.layers.Layer:
        pass

    def get_value(self, td: TrainingDataGeneric, input_layer: keras.layers.Input, sym_raw: dict, X_raw: dict):
        if isinstance(self.feature, (int, float)):
            val = self.feature
            sym = self.feature
        else:
            val, sym = self.feature.get_value(td, input_layer, sym_raw, X_raw)

        return self._get_value(val), self._get_value_symbolic(sym)

    @abstractmethod
    def _get_value(self, val):
        pass

    @abstractmethod
    def _get_value_symbolic(self, sym):
        pass

    def _get_config(self) -> dict:
        c = super()._get_config()
        if isinstance(self.feature, ModularExpression):
            fn = self.feature.name
        else:
            fn = self.feature
        c.update({
            'feature': fn,
        })
        return c

    @classmethod
    def _from_config(cls, item_config: dict, config: list[dict]) -> 'ModularOne':
        """
        Creates a ModularOne instance (or its subclass) from a configuration dictionary.
        Handles reconstruction of the operand modular expression if it was a ModularExpression object.

        Args:
            item_config (dict): Configuration dictionary. Must contain 'feature'.
            config (list[dict]): The list with the configuration dictionaries of all modular expressions

        Returns:
            ModularOne: An instance of the specific ModularOne subclass.
        """

        # Reconstruct feature
        if isinstance(item_config['feature'], dict):
            feature_conf = item_config['feature']
            # Check if modular expression already exists
            fn = ModularExpression.get_existing_modular_expression(feature_conf['name'])
            if fn is None:
                fn = modular_expression_from_config(feature_conf, config)
        elif isinstance(item_config['feature'], str):
            fn = ModularExpression.get_existing_modular_expression(item_config['feature'])
        else:  # feature is int or float
            fn = item_config['feature']
        item_config['feature'] = fn

        return cls(**item_config)


# Numeric (numpy) and symbolic (casadi) implementation of the supported activation functions.
# The keys must be valid keras activations which are also supported by the casadi export of
# AgentLib-MPC, so that training and MPC use the identical function.
ACTIVATION_FUNCTIONS = {
    'softplus': (lambda x: np.logaddexp(0, x), lambda x: ca.log(1 + ca.exp(x))),
    'exponential': (np.exp, ca.exp),
    'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), lambda x: 1 / (1 + ca.exp(-x))),
    'tanh': (np.tanh, ca.tanh),
    'relu': (lambda x: np.maximum(0, x), lambda x: ca.fmax(0, x)),
    'linear': (lambda x: x, lambda x: x),
}


@register_modular_expression
class ModularActivation(ModularOne):
    """
    Applies an activation function (e.g. softplus) elementwise to a modular expression.
    """

    def __init__(self, feature: Union[ModularExpression, int, float], activation: str = 'softplus', name: str = None):
        if activation not in ACTIVATION_FUNCTIONS.keys():
            raise NotImplementedError(f"Activation '{activation}' is not supported. Supported activations are: "
                                      f"{list(ACTIVATION_FUNCTIONS.keys())}")
        if name is None:
            name = f"{activation}({get_name(feature)})"
        super().__init__(feature, name)
        self.activation = activation

    def _construct(self, layer: keras.layers.Layer) -> keras.layers.Layer:
        # The activation is applied by a frozen Dense layer with an identity kernel and without bias,
        # i.e. activation(x @ I) = activation(x). A plain keras.layers.Activation would be simpler, but
        # Dense layers are already supported by the casadi export of AgentLib-MPC.
        units = layer.shape[-1]
        return keras.layers.Dense(units=units, activation=self.activation, use_bias=False,
                                  kernel_initializer=keras.initializers.Identity(), trainable=False)(layer)

    def _get_value(self, val):
        return ACTIVATION_FUNCTIONS[self.activation][0](val)

    def _get_value_symbolic(self, sym):
        return ACTIVATION_FUNCTIONS[self.activation][1](sym)

    def _get_config(self) -> dict:
        c = super()._get_config()
        c.update({
            'activation': self.activation,
        })
        return c


def softplus(feature: Union[ModularExpression, int, float], name: str = None) -> ModularActivation:
    """
    log(1 + exp(feature)) as modular expression.
    """
    return ModularActivation(feature, 'softplus', name)


def exp(feature: Union[ModularExpression, int, float], name: str = None) -> ModularActivation:
    """
    exp(feature) as modular expression.
    """
    return ModularActivation(feature, 'exponential', name)


def sigmoid(feature: Union[ModularExpression, int, float], name: str = None) -> ModularActivation:
    """
    1 / (1 + exp(-feature)) as modular expression.
    """
    return ModularActivation(feature, 'sigmoid', name)


def tanh(feature: Union[ModularExpression, int, float], name: str = None) -> ModularActivation:
    """
    tanh(feature) as modular expression.
    """
    return ModularActivation(feature, 'tanh', name)


def relu(feature: Union[ModularExpression, int, float], name: str = None) -> ModularActivation:
    """
    max(0, feature) as modular expression.
    """
    return ModularActivation(feature, 'relu', name)
