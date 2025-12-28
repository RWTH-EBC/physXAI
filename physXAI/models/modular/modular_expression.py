from abc import ABC, abstractmethod
import os
from typing import Union
from physXAI.models.ann.keras_models.keras_models import ConstantLayer, DivideLayer, InputSliceLayer, PowerLayer
from physXAI.preprocessing.training_data import TrainingDataGeneric
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class ModularExpression(ABC):

    feature_list = dict()
    feature_list_normalized = dict()
    trainable_parameters = dict()
    models = dict()

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def reset():
        ModularExpression.feature_list = dict()
        ModularExpression.feature_list_normalized = dict()
        ModularExpression.trainable_parameters = dict()
        ModularExpression.models = dict()

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
    
    def rename(self, name: str):
        self.name = name

    def get_config(self) -> dict:
        c = {
            'class_name': self.__class__.__name__,
            'name': self.name,
        }
        return c


def get_name(feature: Union[ModularExpression, int, float]) -> str:
    if isinstance(feature, ModularExpression):
        return feature.name
    else:
        return str(feature)


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

    def get_config(self) -> dict:
        c = super().get_config()
        c.update({
            'normalize': self.normalize,
        })
        return c


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

    def get_config(self) -> dict:
        c = super().get_config()
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

    def get_config(self) -> dict:
        c = super().get_config()
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


class ModularAdd(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}+{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return keras.layers.Add()([layer1, layer2])
    

class ModularSub(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}-{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return keras.layers.Subtract()([layer1, layer2])
    

class ModularMul(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}*{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return keras.layers.Multiply()([layer1, layer2])
    

class ModularTrueDiv(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}/{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return DivideLayer()([layer1, layer2])
    

class ModularPow(ModularTwo):

    def __init__(self, feature1: Union[ModularExpression, int, float], feature2: Union[ModularExpression, int, float], name: str = None):
        if name is None:
            name = f"({get_name(feature1)}**{get_name(feature2)})"
        super().__init__(feature1, feature2, name)

    def _construct(self, layer1: keras.layers.Layer, layer2: keras.layers.Layer) -> keras.layers.Layer:
        return PowerLayer()([layer1, layer2])
