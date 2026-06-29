import os
from typing import Optional, Union, List

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

FeatureIndex = Optional[Union[int, List[int]]]

def _inverse_softplus(value: float) -> float:
    
    value = float(value)

    if value <= 0:
        raise ValueError("Initial value must be positive.")

    if value > 20.0:
        return value

    return float(np.log(np.expm1(value)))

@keras.saving.register_keras_serializable(package="custom_layer", name="RC1R1CLayer",)
class RC1R1CLayer(keras.Layer):
    """
    
    """
    def __init__(
        self,
        time_step: float,
        resistance: float,
        capacitance: float,
        theta_solar_init: float,
        t_room_index: int,
        t_ambient_index: int,
        v_flow_ahu_index: FeatureIndex = None,
        t_ahu_sup_index: FeatureIndex = None,
        h_dir_nor_index: FeatureIndex = None,
        q_int_index: FeatureIndex = None,
        use_internal_gains: bool = False,
        trainable_rc: bool = False,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        
        super().__init__(**kwargs)

        if time_step <= 0:
            raise ValueError("time_step must be positive")
        
        if resistance <= 0:
            raise ValueError("resistance must be positive")
        
        if capacitance <= 0:
            raise ValueError("capacitance must be positive")
        
        if theta_solar_init <= 0:
            raise ValueError("theta_solar_init must be positive")
        
        self.time_step = float(time_step)

        self.initial_resistance = float(resistance)
        self.initial_capacitance = float(capacitance)
        self.theta_solar_init = float(theta_solar_init)

        self.t_room_index = int(t_room_index)
        self.t_ambient_index = int(t_ambient_index)

        self.v_flow_ahu_index = v_flow_ahu_index
        self.t_ahu_sup_index = t_ahu_sup_index
        self.h_dir_nor_index = h_dir_nor_index

        self.q_int_index = q_int_index
        self.use_internal_gains = bool(use_internal_gains)
        self.trainable_rc = bool(trainable_rc)

        if self.use_internal_gains and not self.trainable_rc:
            raise ValueError(
                "Configuration error: 'use_internal_gains' can only be True if 'trainable_rc' is also set to True."
            )

        self.epsilon = float(epsilon)

        self.raw_resistance = None
        self.raw_capacitance = None
        self.raw_theta_solar = None

        self.rho_air = 1.204
        self.cp_air = 1005.0

        # Internally mapped parameters and weights
        self.initial_H = None
        self.initial_K = None
        self.opt_h_factor = None
        self.opt_k_factor = None

    def build(self, input_shape):
        """
        
        """
        theta_solar_initializer = _inverse_softplus(self.theta_solar_init)

        self.initial_H = 1.0 / self.initial_resistance
        self.initial_K = 1.0 / self.initial_capacitance

        initializer_val = _inverse_softplus(1.0)

        self.opt_h_factor = self.add_weight(
            name="opt_h_factor",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.opt_k_factor = self.add_weight(
            name="opt_k_factor",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.raw_theta_solar = self.add_weight(
            name="raw_theta_solar",
            shape=(1,),
            initializer=keras.initializers.Constant(theta_solar_initializer),
            trainable=True 
        )

        super().build(input_shape)


    def _positive_parameter(self, raw_parameter):
        """
        
        """
        return keras.activations.softplus(raw_parameter) + self.epsilon
    

    def _take_feature(self, inputs, feature_index: FeatureIndex, reference):
        """
        
        """
        if feature_index is None:
            return keras.ops.zeros_like(reference)
        
        if isinstance(feature_index, int):
            indices = [feature_index]
        else:
            indices = list(feature_index)

        extracted = keras.ops.take(
            inputs,
            indices,
            axis=-1,
        )

        return keras.ops.sum(extracted, axis=-1, keepdims=True)
    

    def call(self, inputs, **kwargs):
        """
        
        """
        t_room = self._take_feature(
            inputs=inputs,
            feature_index=self.t_room_index,
            reference=inputs,
        )

        t_ambient = self._take_feature(
            inputs=inputs,
            feature_index=self.t_ambient_index,
            reference=t_room,
        )

        v_flow_ahu = self._take_feature(
            inputs=inputs, 
            feature_index=self.v_flow_ahu_index, 
            reference=t_room
        )

        t_ahu_sup = self._take_feature(
            inputs=inputs, 
            feature_index=self.t_ahu_sup_index, 
            reference=t_room
        )

        h_dir_nor = self._take_feature(
            inputs=inputs, 
            feature_index=self.h_dir_nor_index, 
            reference=t_room
        )

        h_factor = self._positive_parameter(self.opt_h_factor)
        k_factor = self._positive_parameter(self.opt_k_factor)

        H_phys = h_factor * self.initial_H
        K_phys = k_factor * self.initial_K

        theta_solar = self._positive_parameter(self.raw_theta_solar)

        q_int = self._take_feature(inputs, self.q_int_index, t_room)

        q_transmission = H_phys * (t_ambient - t_room)

        v_flow_m3_s = v_flow_ahu / 3600.0
        q_ahu = self.rho_air * self.cp_air * v_flow_m3_s * (t_ahu_sup - t_room)

        q_solar = theta_solar * h_dir_nor

        q_total = q_transmission + q_ahu + q_solar

        if self.use_internal_gains:
            q_total = q_total + q_int

        delta_t_phys = self.time_step * K_phys * q_total

        return delta_t_phys
    

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)


    def get_config(self):
        """
        
        """
        config = super().get_config()

        config.update(
            {
                "time_step": self.time_step,
                "resistance": self.initial_resistance,
                "capacitance": self.initial_capacitance,
                "theta_solar_init": self.theta_solar_init,
                "t_room_index": self.t_room_index,
                "t_ambient_index": self.t_ambient_index,
                "v_flow_ahu_index": self.v_flow_ahu_index,
                "t_ahu_sup_index": self.t_ahu_sup_index,
                "h_dir_nor_index": self.h_dir_nor_index,
                "q_int_index": self.q_int_index,
                "use_internal_gains": self.use_internal_gains,
                "trainable_rc": self.trainable_rc,
                "epsilon": self.epsilon,
            }
        )

        return config
    


