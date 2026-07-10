import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

@keras.saving.register_keras_serializable(package='custom_layer', name='PhysicsLossLayer')
class PhysicsLossLayer(keras.Layer):
    """
    
    """
    def __init__(
            self,
            weight: float=1.0,
            **kwargs,
    ):
        """
        
        """
        super().__init__(**kwargs)
        self.weight = float(weight)


    def call(self, inputs, **kwargs):
        """
        
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("PhysicsLossLayer expects exactly two inputs: [y_nn, residual].")
        
        y_nn, residual = inputs

        physics_loss = keras.ops.mean(keras.ops.square(residual))

        weighted_physics_loss = self.weight * physics_loss

        self.add_loss(weighted_physics_loss)

        return y_nn
    

    def compute_output_shape(self, input_shape):
        """
        
        """
        return input_shape[0]
    

    def get_config(self):
        """
        
        """
        config = super().get_config()
        config.update({
            'weight': self.weight,
        })
        return config
    
    
    @classmethod
    def from_config(cls, config):
        """
        
        """
        return cls(**config)