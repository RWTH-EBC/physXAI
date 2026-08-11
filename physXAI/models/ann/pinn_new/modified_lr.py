import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

@keras.saving.register_keras_serializable(package="custom_optimizer", name="MultiplierAdam")
class MultiplierAdam(keras.optimizers.Adam):
    """
    
    """
    def __init__(self, lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        if lr_multipliers is not None:
            self.lr_multipliers = lr_multipliers
        else:
            self.lr_multipliers = {} 

    def update_step(self, gradient, variable, learning_rate):
        modified_lr = learning_rate

        variable_name = getattr(variable, "path", variable.name)

        # parameter_name = variable_name.split("/")[-1]

        # if parameter_name in self.lr_multipliers:
        #     modified_lr = learning_rate * self.lr_multipliers[parameter_name]

        for mult_name in self.lr_multipliers:
            if mult_name in variable_name:
                modified_lr = learning_rate * self.lr_multipliers[mult_name]
                break

        super().update_step(gradient, variable, modified_lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr_multipliers": self.lr_multipliers
            }
        )

        return config