import os
from typing import Union
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
from keras.saving import serialize_keras_object, deserialize_keras_object

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


@keras.saving.register_keras_serializable(package='custom_layer', name='InputSliceLayer')
class InputSliceLayer(keras.Layer):
    """
    A simple layer to select specific features from the last axis.
    """

    def __init__(self, feature_indices: Union[int, list[int]], **kwargs):
        """
        Initializes the layer.

        Args:
            feature_indices (int or list): The index or indices to select.
                - If int (e.g., 1), selects the feature at that index and
                  reduces the rank.
                - If list (e.g., [1]), selects the feature(s) and
                  keeps the rank.
        """
        super().__init__(**kwargs)
        self.feature_indices = feature_indices

    def call(self, inputs):
        return keras.ops.take(inputs, self.feature_indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_indices": self.feature_indices
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        if isinstance(self.feature_indices, int):
            output_shape.pop(-1)

        elif isinstance(self.feature_indices, (list, tuple)):
            output_shape[-1] = len(self.feature_indices)

        return tuple(output_shape)


@keras.saving.register_keras_serializable(package='custom_layer', name='ConstantLayer')
class ConstantLayer(keras.Layer):
    """
    A layer that returns a constant tensor, broadcasted to the batch size.

    This layer ignores its input and simply returns a tensor of a
    pre-defined shape, initialized with a constant value.

    The constant is created as a Keras weight, which can be
    trainable or non-trainable.
    """

    def __init__(self, value=0.0, shape=(1,), trainable=False, weight_name: str = None, **kwargs):
        """
        Initializes the layer.

        Args:
            initial_value (float): The value to initialize the constant tensor with.
            shape (tuple): The shape of the constant, *excluding* the batch
                dimension. For a single number to be added, use (1,).
            trainable (bool): Whether this constant is a learnable parameter.
        """
        super().__init__(trainable=trainable, **kwargs)
        self.value = value
        self.target_shape = tuple(shape)
        self.weight_name = weight_name

    def build(self, input_shape):
        if self.value is not None:
            init = keras.initializers.Constant(self.value)
        else:
            init = keras.initializers.glorot_uniform()
        self.constant = self.add_weight(
            shape=self.target_shape,
            initializer=init,
            trainable=self.trainable,
            name=self.weight_name,
        )

    def call(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]

        # Create the full target shape, including the batch dimension
        # e.g., (batch_size,) + (1,) -> (batch_size, 1)
        full_shape = (batch_size,) + self.target_shape

        return keras.ops.broadcast_to(self.constant, full_shape)

    def compute_output_shape(self, input_shape):
        # The output shape is (batch_size,) + our target_shape
        return (input_shape[0],) + self.target_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "value": self.value,
            "shape": self.target_shape,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='custom_layer', name='DivideLayer')
class DivideLayer(keras.Layer):
    """
    A layer that divides two layers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return keras.ops.divide(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='custom_layer', name='PowerLayer')
class PowerLayer(keras.Layer):
    """
    A layer that computes the power of two layers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return keras.ops.power(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='custom_layer', name='SliceFeatures')
class SliceFeatures(keras.Layer):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[..., self.start:self.end]

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


@keras.saving.register_keras_serializable(package='custom_constraint', name='DiagonalConstraint')
class DiagonalPosConstraint(keras.constraints.Constraint):
    """Keep only the diagonal of a 2D weight matrix (intended for RNN recurrent kernels)."""

    def __call__(self, w):
        # Convert to Keras tensor
        # w = keras.ops.convert_to_tensor(w)

        # Dynamic shape: (rows, cols)
        shape = keras.ops.shape(w)
        rows, cols = shape[0], shape[1]

        # Runtime check for squareness
        if rows != cols:
            raise ValueError(
                f"DiagonalConstraint expects a square matrix, got shape {w.shape}"
            )

        # Build an identity mask
        mask = keras.ops.eye(rows, cols, dtype=w.dtype)

        # Keep only diagonal elements
        w = w * mask

        # Extract diagonal values
        diag = keras.ops.diagonal(w)

        # Set negative diagonal values to zero
        diag_clipped = keras.ops.maximum(diag, 0)

        # Rebuild matrix with non-negative diagonal
        # Create a matrix where only the diagonal is diag_clipped
        w = keras.ops.eye(rows, cols, dtype=w.dtype) * keras.ops.expand_dims(diag_clipped, 0)

        return w

    def get_config(self):
        return {}

@keras.saving.register_keras_serializable(package='custom_cell', name='PCNNCell')
class PCNNCell(keras.Layer):
    def __init__(self, dis_ann: keras.models, dis_inputs: int,
                 non_lin_ann: keras.models = None, non_lin_inputs: int = None, **kwargs):
        super(PCNNCell, self).__init__(**kwargs)

        # Define layers for ANN and Linear modules
        self.dis_ann = dis_ann
        self.dis_inputs = dis_inputs
        self.lin_layer = keras.layers.Dense(1, activation='linear', kernel_constraint=keras.constraints.NonNeg(), name='lin_layer')  # has to be trainable!
        self.non_lin_ann = non_lin_ann
        self.non_lin_inputs = non_lin_inputs

        # instantiate add and concatenate layer here to use it in call
        self.add_layer = keras.layers.Add(trainable=False)
        self.concatenate_layer = keras.layers.Concatenate(trainable=False)

    @property
    def state_size(self):
        return [2]  # Return list with sizes of D_k+1 and E_k+1

    @property
    def output_size(self):
        return 1

    def build(self, input_shape):
        super(PCNNCell, self).build(input_shape)

        lin_input_shape = (input_shape[0], input_shape[1]-self.dis_inputs-self.non_lin_inputs+1)
        self.lin_layer.build(lin_input_shape)

    def call(self, inputs, states):
        # TODO Evtl. hier auch cropping nötig
        states = states[0]  # states is a tuple of tensors, therefore get first element of tuple
        previous_state_D = states[:, 0]  # Previous state D_k+1
        previous_state_D = keras.ops.reshape(previous_state_D, (-1, 1))
        previous_state_E = states[:, 1]  # Previous state E_k+1
        previous_state_E = keras.ops.reshape(previous_state_E, (-1, 1))

        # disturbance module
        disturbance_inputs = inputs[:, :self.dis_inputs]  # evtl. als CroppingLayer (evtl. ist der nur für cropping auf time axis) or Reshape First part for ANN module
        dis_ann_output = self.dis_ann(disturbance_inputs)

        # linear module
        if self.non_lin_inputs is None:
            linear_inputs = inputs[:, self.dis_inputs:]
            lin_module_output = self.lin_layer(linear_inputs)

        else:
            # non-linear inputs have to be fed through additional ANN to capture
            # non-linear dynamics appropriately before entering lin module
            non_linear_inputs = inputs[:, -self.non_lin_inputs:]
            non_linear_output = self.non_lin_ann(non_linear_inputs)

            linear_inputs = inputs[:, self.dis_inputs:-self.non_lin_inputs]
            lin_module_inputs = self.concatenate_layer([linear_inputs, non_linear_output])
            lin_module_output = self.lin_layer(lin_module_inputs)

        # State D_k+1 is output of disturbance ann + previous state
        D_k_plus_1 = self.add_layer([previous_state_D, dis_ann_output])
        # State E_k+1 is output of linear module + previous state
        E_k_plus_1 = self.add_layer([previous_state_E, lin_module_output])
        T_k_plus_1 = self.add_layer([D_k_plus_1, E_k_plus_1])

        return T_k_plus_1, [tf.concat([D_k_plus_1, E_k_plus_1], axis=1)]  # Return output and updated state

    def get_config(self):
        config = {
            "dis_ann": serialize_keras_object(self.dis_ann),
            "dis_inputs": self.dis_inputs,
            "non_lin_ann": serialize_keras_object(self.non_lin_ann),
            "non_lin_inputs": self.non_lin_inputs
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # Pull out serialized specs
        dis_ann_cfg = config.pop("dis_ann")
        non_lin_ann_cfg = config.pop("non_lin_ann")

        # Explicitly rebuild objects
        dis_ann = deserialize_keras_object(dis_ann_cfg) if dis_ann_cfg is not None else None
        non_lin_ann = deserialize_keras_object(non_lin_ann_cfg) if dis_ann_cfg is not None else None

        # Pass remaining simple fields to __init__
        obj = cls(dis_ann=dis_ann, non_lin_ann=non_lin_ann, **config)
        return obj
