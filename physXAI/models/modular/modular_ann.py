from logging import warning
import os
from typing import Optional, Union

from physXAI.models.modular.modular_expression import ModularExpression
from physXAI.models.ann.ann_design import ANNModel, CMNNModel, ClassicalANNModel
from physXAI.models.models import register_model
from physXAI.preprocessing.training_data import TrainingDataGeneric
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


@register_model
class ModularANN(ANNModel):
    """
    A Modular Artificial Neural Network model that allows for custom architectures.
    """

    def __init__(self, architecture: ModularExpression, batch_size: int = 32, epochs: int = 1000,
                 learning_rate: float = 0.001, early_stopping_epochs: Optional[int] = 100,
                 random_seed: int = 42, **kwargs):
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
        """

        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed)
        self.architecture: ModularExpression = architecture

        self.model_config.update({})

    def generate_model(self, **kwargs):
        """
        Generates the Keras model using the specified modular architecture.
        """

        td = kwargs['td']
        n_features = td.X_train_single.shape[1]
        input_layer = keras.layers.Input(shape=(n_features,))
        x = self.architecture.construct(input_layer, td)
        model = keras.models.Model(inputs=input_layer, outputs=x)
        model.summary()
        return model

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({})
        warning("ModularANN currently does not save architecture config.")
        return config


class ModularModel(ModularExpression):

    allowed_models = [ClassicalANNModel, CMNNModel]
    i = 0

    def __init__(self, model: ANNModel, inputs: list[ModularExpression], rescale_output: bool = False, name: str = None):
        if not any(isinstance(model, allowed) for allowed in self.allowed_models):
            raise NotImplementedError(f"Currently {type(model)} is not supported. Allowed models are: {self.allowed_models}")

        if name is None:
            name = f"ModularModel_{ModularModel.i}"
            ModularModel.i += 1

        super().__init__(name)
        self.model = model
        self.rescale_output = rescale_output
        if rescale_output:
            warning("Using rescale_output=True in ModularANN should only be done if model output is training data output.")
        self.model.model_config.update({
            "normalize": False,
            "rescale_output": rescale_output
        })
        self.inputs = inputs

    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)
            self.model.model_config['n_features'] = len(inps)
            td.columns = [inp.name for inp in self.inputs]
            l = self.model.generate_model(td=td)(keras.layers.Concatenate()(inps))
            ModularExpression.models[self.name] = l
            return l    


class ModularLinear(ModularExpression):
    i = 0

    def __init__(self, inputs: list[ModularExpression]):
        if name is None:
            name = f"ModularLinear_{ModularLinear.i}"
            ModularLinear.i += 1
        super().__init__(name)
        
    def construct(self, input_layer: keras.layers.Input, td: TrainingDataGeneric) -> keras.layers.Layer:
        if self.name in ModularExpression.models.keys():
            return ModularExpression.models[self.name]
        else:
            inps = list()
            for x in self.inputs:
                y = x.construct(input_layer, td)
                inps.append(y)
            l = keras.layers.Dense(units=1, activation='linear')(keras.layers.Concatenate()(inps))
            ModularExpression.models[self.name] = l
            return l