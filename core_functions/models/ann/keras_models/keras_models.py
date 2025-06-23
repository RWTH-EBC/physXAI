import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


@keras.saving.register_keras_serializable(package='custom_constraint', name='NonNegPartial')
class NonNegPartial(keras.constraints.Constraint):
    """
    A Keras constraint that enforces non-negativity or non-positivity on specific parts of a weight tensor.
    This is useful for imposing monotonicity constraints on a neural network layer.
    For example, if a feature should have a non-decreasing relationship with the output,
    the corresponding weight can be constrained to be non-negative.
    """

    def __init__(self, monotonicities: list[int]):
        """
        Initializes the NonNegPartial constraint.

        Args:
            monotonicities (list[int]): A list of integers specifying the monotonicity for each
                                        corresponding weight (or part of the weight tensor).
                                        -  1: Enforces non-negativity (weight >= 0).
                                        - -1: Enforces non-positivity (weight <= 0).
                                        -  0: No constraint is applied.
        Raises:
            ValueError: If any item in `monotonicities` is not -1, 0, or 1.
        """

        allowed_items = [-1, 0, 1]
        if not all(item in allowed_items for item in monotonicities):
            raise ValueError('Monotonicities must be in [-1, 0, 1]')
        self.monotonicities: list[int] = monotonicities

    def __call__(self, w):
        """
         Applies the constraint to the weight tensor.
         This method is called by Keras during the training process after each weight update.

         Args:
             w: The weight tensor to be constrained.

         Returns:
             The constrained weight tensor.

         Raises:
             ValueError: If the length of `monotonicities` does not match the first dimension
                         of the weight tensor `w`.
         """

        w = keras.ops.convert_to_tensor(w)

        if len(self.monotonicities) != w.shape[0]:
            raise ValueError('Length of monotonicities list must be equal'
                             ' to the first element of the weight tensor´s shape.')

        w_split = keras.ops.split(w, w.shape[0])
        for i in range(0, w.shape[0]):
            # non-negativity
            if self.monotonicities[i] == 1:
                w_split[i] = w_split[i] * keras.ops.cast(keras.ops.greater_equal(w_split[i], 0.),
                                                         dtype=w_split[i].dtype)
            # non - positivity
            elif self.monotonicities[i] == -1:
                w_split[i] = w_split[i] * keras.ops.cast(keras.ops.greater_equal(-w_split[i], 0.),
                                                         dtype=w_split[i].dtype)
            else:
                continue

        return keras.ops.concatenate(w_split)

    def get_config(self):
        return {'monotonicities': self.monotonicities}


@keras.saving.register_keras_serializable(package='custom_activation', name='ConcaveActivation')
class ConcaveActivation:
    """
    A Keras activation function wrapper that transforms a given activation function
    into its concave counterpart.
    If f(x) is the original activation, the concave version is -f(-x).
    """

    def __init__(self, activation: str):
        """
        Initializes the ConcaveActivation.

        Args:
            activation (str): The name of the Keras activation function to be made concave
                              (e.g., 'relu', 'sigmoid').
        """

        self.activation = activation
        self.activation_fcn = keras.activations.get(activation)

    def __call__(self, x):
        """
        Applies the concave transformation to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The tensor after applying the concave activation: -activation_fcn(-x).
        """

        return -self.activation_fcn(-x)

    def get_config(self):
        return {'activation': self.activation}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='custom_activation', name='SaturatedActivation')
class SaturatedActivation:
    """
    A Keras activation function that creates a saturated version of a given base activation.
    The saturation behavior is different for x <= 0 and x > 0.
    - For x <= 0:  f(x + 1) - f(1)  (where f is the base activation)
    - For x > 0:  g(x - 1) + f(1)  (where g is the concave version of f, and f(1) is a constant)
    This can be used to create activation functions that plateau or saturate at certain input ranges.
    """

    def __init__(self, activation: str):
        """
        Initializes the SaturatedActivation.

        Args:
            activation (str): The name of the Keras activation function to be used as the base.
        """

        self.activation = activation
        self.activation_fcn = keras.activations.get(activation)
        self.activation_fcn_concave = ConcaveActivation(activation)

    def __call__(self, x):
        """
        Applies the saturated activation to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The tensor after applying the saturated activation.
        """

        cc = self.activation_fcn(keras.ops.ones_like(x))
        return keras.ops.where(
            x <= 0,
            self.activation_fcn(x + 1) - cc,
            self.activation_fcn_concave(x - 1) + cc,
        )

    def get_config(self):
        return {'activation': self.activation}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='custom_activation', name='LimitedActivation')
class LimitedActivation:
    """
    A Keras activation function that clips the input tensor to a specified minimum and/or maximum value.
    """

    def __init__(self, max_value: float = None, min_value: float = None):
        """
        Initializes the LimitedActivation.

        Args:
            max_value (float, optional): The maximum value to clip to. If None, no upper limit is applied.
                                         Defaults to None.
            min_value (float, optional): The minimum value to clip to. If None, no lower limit is applied.
                                         Defaults to None.
        """

        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, x):
        """
        Applies the clipping to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The clipped tensor.
        """

        if self.min_value is not None:
            x = keras.ops.maximum(x, self.min_value)
        if self.max_value is not None:
            x = keras.ops.minimum(x, self.max_value)
        return x

    def get_config(self):
        return {'max_value': self.max_value, 'min_value': self.min_value}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='custom_layer', name='RBFLayer')
class RBFLayer(keras.Layer):
    """
        Custom Radial Basis Function (RBF) Layer.

        This layer implements RBF neurons, where the activation is typically a Gaussian function
        of the Euclidean distance between the input and the neuron's center.

        Arguments:
            units (int): Positive integer, dimensionality of the output space (number of RBF neurons).
            gamma (float or list/array): The gamma parameter of the Gaussian function, controlling the width.
                                         Can be a scalar (same gamma for all neurons) or a tensor/array
                                         of length `units` (individual gamma per neuron).
            initial_centers (np.ndarray, optional): A NumPy array of shape (units, input_dim)
                                                     for the initial centers. If None, they are
                                                     initialized using a default initializer (RandomUniform).
            learnable_centers (bool): Whether the centers should be trainable. Defaults to True.
            learnable_gamma (bool): Whether gamma should be trainable. Defaults to True.

        Input shape:
            2D tensor with shape `(batch_size, input_dim)`.

        Output shape:
            2D tensor with shape `(batch_size, units)`.
    """

    def __init__(self, units, gamma=1.0, initial_centers=None,
                 learnable_centers=True, learnable_gamma=True, **kwargs):
        """
        Initializes the RBFLayer.

        Args:
            units (int): Number of RBF neurons.
            gamma (float or list/np.ndarray): Initial value(s) for the gamma parameter.
            initial_centers (np.ndarray, optional): Initial positions for the RBF centers.
            learnable_centers (bool): If True, centers will be updated during training.
            learnable_gamma (bool): If True, gamma values will be updated during training.
        """

        super().__init__(**kwargs)
        self.units = units
        self.gamma_init_value = gamma
        self.initial_centers = initial_centers
        self.learnable_centers = learnable_centers
        self.learnable_gamma = learnable_gamma

        self.input_dim = None
        self.centers = None
        self.log_gamma = None

    def build(self, input_shape):
        """
        Creates the layer's weights (centers and gamma).
        This method is called the first time the layer is used, with the shape of the input.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """

        # Extract the input feature dimension
        self.input_dim = input_shape[-1]

        # Initialize RBF centers
        if self.initial_centers is not None:
            # Validate the shape of provided initial centers
            if self.initial_centers.shape != (self.units, self.input_dim):
                raise ValueError(
                    f"Shape of initial_centers {self.initial_centers.shape} "
                    f"does not match expected shape ({self.units}, {self.input_dim})"
                )
            centers_initializer = keras.initializers.Constant(self.initial_centers)
        else:
            # Default initializer for centers if none are provided (RandomUniform)
            centers_initializer = keras.initializers.RandomUniform(minval=0., maxval=1.)

        # Add centers as a trainable weight to the layer
        self.centers = self.add_weight(
            name='centers',
            shape=(self.units, self.input_dim),
            initializer=centers_initializer,
            trainable=self.learnable_centers
        )

        # Initialize gamma parameters (width of the Gaussian function)
        # We store and train log_gamma to ensure gamma = exp(log_gamma) remains positive.
        if isinstance(self.gamma_init_value, (list, np.ndarray)):
            # If gamma is provided as a list or array, it's for individual neurons
            if len(self.gamma_init_value) != self.units:
                raise ValueError("If gamma is a list/array, its length must be equal to units.")
            # Convert initial gamma values to log_gamma
            initial_log_gamma = np.log(self.gamma_init_value).astype(np.float32)
        else:
            # If gamma is a scalar, use the same value for all neurons
            # Convert initial gamma values to log_gamma
            initial_log_gamma = np.full(self.units, np.log(self.gamma_init_value), dtype=np.float32)

        # Add log_gamma as a trainable weight
        self.log_gamma = self.add_weight(
            name='log_gamma',
            shape=(self.units,),
            initializer=keras.initializers.Constant(initial_log_gamma),
            trainable=self.learnable_gamma
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Defines the forward pass of the RBF layer.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, units), representing
                       the activation of each RBF neuron for each input sample.
        """
        # inputs shape: (batch_size, input_dim)
        # centers shape: (units, input_dim)
        # Goal: Calculate ||inputs_batch_item - center_unit||^2 for all combinations

        # Expand dimensions of inputs and centers to enable broadcasting for distance calculation
        # inputs_expanded shape: (batch_size, 1, input_dim)
        inputs_expanded = keras.ops.expand_dims(inputs, axis=1)
        # centers_expanded shape: (1, units, input_dim)
        centers_expanded = keras.ops.expand_dims(self.centers, axis=0)

        # Calculate squared Euclidean distances between each input sample and each RBF center
        # (inputs_expanded - centers_expanded) results in shape (batch_size, units, input_dim)
        # Then, sum the squares along the input_dim axis (axis=2)
        distances_sq = keras.ops.sum(
            keras.ops.square(inputs_expanded - centers_expanded), axis=2
        )   # Resulting shape: (batch_size, units)

        # Apply the Gaussian RBF activation function: exp(-gamma * ||x - c||^2)
        # Retrieve gamma from log_gamma (shape: (units,))
        gamma = keras.ops.exp(self.log_gamma)
        # Broadcasting will apply each gamma to its respective column in distances_sq
        # Output shape: (batch_size, units)
        phi = keras.ops.exp(-gamma * distances_sq)
        return phi

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor (batch_size, units).
        """

        return input_shape[0], self.units

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            # Store the original gamma value(s), not log_gamma, for easier interpretatio
            "gamma": np.exp(self.log_gamma.numpy()).tolist() if self.log_gamma is not None else self.gamma_init_value,
            # Convert initial centers to list
            "initial_centers": self.initial_centers.tolist() if isinstance(self.initial_centers, np.ndarray) else None,
            "learnable_centers": self.learnable_centers,
            "learnable_gamma": self.learnable_gamma
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Retrieve 'initial_centers' from config and convert back to NumPy array if it was stored as a list
        initial_centers_list = config.pop("initial_centers", None)
        if initial_centers_list is not None:
            config["initial_centers"] = np.array(initial_centers_list)
        return cls(**config)
