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
            reduction: str = 'mean',
            **kwargs,
    ):
        """
        
        """
        super().__init__(**kwargs)

        if reduction not in ['mean', 'sum']:
            raise ValueError("Unsupported reduction. Use 'mean' or 'sum'.")
        
        self.weight = float(weight)
        self.reduction = reduction


    def call(self, inputs, **kwargs):
        """
        
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("PhysicsLossLayer expects exactly two inputs: [y_nn, residual].")
        
        y_nn, residual = inputs

        squared_residual = keras.ops.square(residual)

        if self.reduction == 'mean':
            physics_loss = keras.ops.mean(squared_residual)
        else:
            physics_loss = keras.ops.sum(squared_residual)

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
            'reduction': self.reduction,
        })
        return config
    
    
    @classmethod
    def from_config(cls, config):
        """
        
        """
        return cls(**config)