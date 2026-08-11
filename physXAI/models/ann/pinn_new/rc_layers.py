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

def _inverse_sigmoid(value: float, epsilon: float = 1e-6) -> float:
    value = float(value)
    value = min(max(value, epsilon), 1.0 - epsilon)
    return float(np.log(value / (1.0 - value)))


@keras.saving.register_keras_serializable(package="custom_layer", name="RC1R1CLayer",)
class RC1R1CLayer(keras.Layer):
    """
    
    """
    def __init__(
        self,
        time_step: float,
        r_win: float,
        r_ext: float,
        c_air: float,
        t_air_index: int,
        t_amb_index: int,
        theta_solar_init: float = 1.75,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        v_flow_ahu_index: FeatureIndex = None,
        t_ahu_sup_index: FeatureIndex = None,
        t_sup_w_h_index: FeatureIndex = None,
        y_valve_h_index: FeatureIndex = None,
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
        
        if r_win <= 0:
            raise ValueError("resistance of the window must be positive")
        
        if r_ext <= 0:
            raise ValueError("resistance of the wall must be positive")
        
        if c_air <= 0:
            raise ValueError("capacitance of the air must be positive")
        
        if theta_solar_init <= 0:
            raise ValueError("theta_solar_init must be positive")
        
        self.time_step = float(time_step)

        self.initial_r_win = float(r_win)
        self.initial_r_ext = float(r_ext)
        self.initial_c_air = float(c_air)
        self.theta_solar_init = float(theta_solar_init)
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        
        self.initial_tau_ext_air = self.initial_r_ext * self.initial_c_air
        self.initial_kappa_ext_air = 1.0 / self.initial_tau_ext_air
        self.initial_tau_win_air = self.initial_r_win * self.initial_c_air
        self.initial_kappa_win_air = 1.0 / self.initial_tau_win_air
        self.initial_k_air = 1.0 / self.initial_c_air

        self.t_air_index = int(t_air_index)
        self.t_amb_index = int(t_amb_index)

        self.v_flow_ahu_index = v_flow_ahu_index
        self.t_ahu_sup_index = t_ahu_sup_index
        self.t_sup_w_h_index = t_sup_w_h_index
        self.y_valve_h_index = y_valve_h_index
        self.h_dir_nor_index = h_dir_nor_index
        self.q_int_index = q_int_index

        self.use_internal_gains = bool(use_internal_gains)
        self.trainable_rc = bool(trainable_rc)

        if self.use_internal_gains and not self.trainable_rc:
            raise ValueError("Configuration error: 'use_internal_gains' can only be True if 'trainable_rc' is also set to True.")

        self.epsilon = epsilon

        self.rho_air = 1.204
        self.cp_air = 1005.0
        self.V_flow_w_h_max = 100
        self.valve_a = 3.2
        self.m_QT = 0.4464068811
        self.m_QT2 = -0.0003313083
        self.KQ_w1 = 0.0199019187
        self.KQ_w1a1 = -0.0000048959
        self.KQ_w2 = -0.0001255783
        self.KQ_w1a2 = -0.0000000062
        self.KQ_w2a1 = 0.0000000597
        self.KQ_w3 = 0.0000002721

    def build(self, input_shape):
        """
        
        """
        theta_solar_initializer = _inverse_softplus(self.theta_solar_init)
        initializer_val = _inverse_softplus(1.0)

        alpha_initializer = _inverse_sigmoid(self.alpha_init)
        beta_initializer = _inverse_sigmoid(self.beta_init)

        self.opt_kappa_factor_win_air = self.add_weight(
            name="opt_kappa_factor_win_air",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.opt_kappa_factor_ext_air = self.add_weight(
            name="opt_kappa_factor_ext_air",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.opt_k_factor_air = self.add_weight(
            name="opt_k_factor_air",
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

        self.opt_alpha = self.add_weight(
            name="opt_alpha",
            shape=(1,),
            initializer=keras.initializers.Constant(alpha_initializer),
            trainable=True
        )

        self.opt_beta = self.add_weight(
            name="opt_beta",
            shape=(1,),
            initializer=keras.initializers.Constant(beta_initializer),
            trainable=self.use_internal_gains
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
        t_air = self._take_feature(
            inputs=inputs,
            feature_index=self.t_air_index,
            reference=inputs,
        )

        t_ambient = self._take_feature(
            inputs=inputs,
            feature_index=self.t_amb_index,
            reference=t_air,
        )

        v_flow_ahu = self._take_feature(
            inputs=inputs, 
            feature_index=self.v_flow_ahu_index, 
            reference=t_air
        )

        t_ahu_sup = self._take_feature(
            inputs=inputs, 
            feature_index=self.t_ahu_sup_index, 
            reference=t_air
        )

        t_sup_w_h = self._take_feature(
            inputs=inputs,
            feature_index=self.t_sup_w_h_index,
            reference=t_air
        )

        y_valve_h = self._take_feature(
            inputs=inputs,
            feature_index=self.y_valve_h_index,
            reference=t_air
        )

        h_dir_nor = self._take_feature(
            inputs=inputs, 
            feature_index=self.h_dir_nor_index, 
            reference=t_air
        )

        q_int = self._take_feature(
            inputs=inputs, 
            feature_index=self.q_int_index, 
            reference=t_air
        )


        kappa_factor_win_air = self._positive_parameter(self.opt_kappa_factor_win_air)
        kappa_factor_ext_air = self._positive_parameter(self.opt_kappa_factor_ext_air)
        k_factor_air = self._positive_parameter(self.opt_k_factor_air)

        kappa_phys_win_air = kappa_factor_win_air * self.initial_kappa_win_air
        kappa_phys_ext_air = kappa_factor_ext_air * self.initial_kappa_ext_air
        k_phys_air = k_factor_air * self.initial_k_air

        theta_solar = self._positive_parameter(self.raw_theta_solar)
        alpha = keras.activations.sigmoid(self.opt_alpha)
        beta = keras.activations.sigmoid(self.opt_beta)

        term_transmission_ext = kappa_phys_ext_air * (t_ambient - t_air)
        term_transmission_win = kappa_phys_win_air * (t_ambient - t_air)

        v_flow_m3_s = v_flow_ahu / 3600.0
        q_ahu = self.rho_air * self.cp_air * v_flow_m3_s * (t_ahu_sup - t_air)
        term_ahu = k_phys_air * q_ahu

        V_flow_w_h = self.V_flow_w_h_max * (keras.ops.exp(self.valve_a * y_valve_h / 100) - 1) / (keras.ops.exp(self.valve_a) - 1)
        QT = self.m_QT * v_flow_ahu + self.m_QT2 * v_flow_ahu**2
        KQ =  self.KQ_w3 * V_flow_w_h**3 + self.KQ_w2 * V_flow_w_h**2 + self.KQ_w1 * V_flow_w_h + self.KQ_w1a1 * v_flow_ahu * V_flow_w_h + self.KQ_w1a2 * v_flow_ahu**2 * V_flow_w_h + self.KQ_w2a1 * v_flow_ahu * V_flow_w_h**2
        q_did = QT * (t_sup_w_h - t_air) * KQ / (1 + QT * 0.86 / (2*V_flow_w_h + 1))
        term_did = k_phys_air * q_did

        term_solar = alpha * k_phys_air * theta_solar * h_dir_nor

        term_total = term_transmission_ext + term_transmission_win + term_ahu + term_solar + term_did

        if self.use_internal_gains:
            term_total = term_total + beta * k_phys_air * q_int

        delta_t_phys = self.time_step * term_total

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
                "r_win": self.initial_r_win,
                "r_ext": self.initial_r_ext,
                "c_air": self.initial_c_air,
                "theta_solar_init": self.theta_solar_init,
                "alpha_init": self.alpha_init,
                "beta_init": self.beta_init,
                "t_air_index": self.t_air_index,
                "t_amb_index": self.t_amb_index,
                "v_flow_ahu_index": self.v_flow_ahu_index,
                "t_ahu_sup_index": self.t_ahu_sup_index,
                "t_sup_w_h_index": self.t_sup_w_h_index,
                "y_valve_h_index": self.y_valve_h_index,
                "h_dir_nor_index": self.h_dir_nor_index,
                "q_int_index": self.q_int_index,
                "use_internal_gains": self.use_internal_gains,
                "trainable_rc": self.trainable_rc,
                "epsilon": self.epsilon,
            }
        )

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

@keras.saving.register_keras_serializable(package="custom_layer", name="RC2R2CPhysNetLayer",)
class RC2R2CPhysNetLayer(keras.Layer):
    def __init__(
        self,
        time_step: float,
        r_win: float,
        r_ext: float,
        c_air: float,
        t_air_index: int,
        t_amb_index: int,
        theta_solar_init: float = 1.75,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        v_flow_ahu_index: FeatureIndex = None,
        t_ahu_sup_index: FeatureIndex = None,
        t_sup_w_h_index: FeatureIndex = None,
        y_valve_h_index: FeatureIndex = None,
        h_dir_nor_index: FeatureIndex = None,
        q_int_index: FeatureIndex = None,
        predict_delta: bool = True,
        use_internal_gains: bool = False,
        trainable_rc: bool = False,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if time_step <= 0:
            raise ValueError("time_step must be positive")
        
        if r_win <= 0:
            raise ValueError("resistance of the window must be positive")
        
        if r_ext <= 0:
            raise ValueError("resistance of the wall must be positive")
        
        if c_air <= 0:
            raise ValueError("capacitance of the air must be positive")
        
        if theta_solar_init <= 0:
            raise ValueError("theta_solar_init must be positive")
        
        self.time_step = float(time_step)

        self.initial_r_win = float(r_win)
        self.initial_r_ext = float(r_ext)
        self.initial_c_air = float(c_air)
        self.theta_solar_init = float(theta_solar_init)
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)

        self.initial_tau_win_air = self.initial_r_win * self.initial_c_air
        self.initial_kappa_win_air = 1.0 / self.initial_tau_win_air
        self.initial_tau_ext_air = self.initial_r_ext * self.initial_c_air
        self.initial_k_air = 1.0 / self.initial_c_air

        self.t_air_index = int(t_air_index)
        self.t_amb_index = int(t_amb_index)

        self.v_flow_ahu_index = v_flow_ahu_index
        self.t_ahu_sup_index = t_ahu_sup_index
        self.t_sup_w_h_index = t_sup_w_h_index
        self.y_valve_h_index = y_valve_h_index
        self.h_dir_nor_index = h_dir_nor_index
        self.q_int_index = q_int_index

        self.predict_delta = bool(predict_delta)
        self.use_internal_gains = bool(use_internal_gains)
        self.trainable_rc = bool(trainable_rc)

        if self.use_internal_gains and not self.trainable_rc:
            raise ValueError("Configuration error: 'use_internal_gains' can only be True if 'trainable_rc' is also set to True.")
        
        self.epsilon = float(epsilon)

        self.rho_air = 1.204
        self.cp_air = 1005.0
        self.V_flow_w_h_max = 100
        self.valve_a = 3.2
        self.m_QT = 0.4464068811
        self.m_QT2 = -0.0003313083
        self.KQ_w1 = 0.0199019187
        self.KQ_w1a1 = -0.0000048959
        self.KQ_w2 = -0.0001255783
        self.KQ_w1a2 = -0.0000000062
        self.KQ_w2a1 = 0.0000000597
        self.KQ_w3 = 0.0000002721


    def build(self, input_shape):
        """
        
        """
        theta_solar_initializer = _inverse_softplus(self.theta_solar_init)
        initializer_val = _inverse_softplus(1.0)

        alpha_initializer = _inverse_sigmoid(self.alpha_init)
        beta_initializer = _inverse_sigmoid(self.beta_init)

        self.opt_kappa_factor_win_air = self.add_weight(
            name="opt_kappa_factor_win_air",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.opt_tau_factor_ext_air = self.add_weight(
            name="opt_tau_factor_ext_air",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.opt_k_factor_air = self.add_weight(
            name="opt_k_factor_air",
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

        self.opt_alpha = self.add_weight(
            name="opt_alpha",
            shape=(1,),
            initializer=keras.initializers.Constant(alpha_initializer),
            trainable=True
        )

        self.opt_beta = self.add_weight(
            name="opt_beta",
            shape=(1,),
            initializer=keras.initializers.Constant(beta_initializer),
            trainable=self.use_internal_gains
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
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("RC2R2CPhysNetlayer expects [x, y_air_pred]!")

        x, y_air_pred = inputs

        t_air = self._take_feature(
            inputs=x,
            feature_index=self.t_air_index,
            reference=x,
        )

        t_amb = self._take_feature(
            inputs=x,
            feature_index=self.t_amb_index,
            reference=t_air,
        )

        v_flow_ahu = self._take_feature(
            inputs=x, 
            feature_index=self.v_flow_ahu_index, 
            reference=t_air
        )

        t_ahu_sup = self._take_feature(
            inputs=x, 
            feature_index=self.t_ahu_sup_index, 
            reference=t_air
        )

        t_sup_w_h = self._take_feature(
            inputs=x,
            feature_index=self.t_sup_w_h_index,
            reference=t_air
        )

        y_valve_h = self._take_feature(
            inputs=x,
            feature_index=self.y_valve_h_index,
            reference=t_air
        )

        h_dir_nor = self._take_feature(
            inputs=x, 
            feature_index=self.h_dir_nor_index, 
            reference=t_air
        )

        
        q_int = self._take_feature(
            inputs=x, 
            feature_index=self.q_int_index, 
            reference=t_air
        )

        kappa_factor_win_air = self._positive_parameter(self.opt_kappa_factor_win_air)
        tau_factor_ext_air = self._positive_parameter(self.opt_tau_factor_ext_air)
        k_factor_air = self._positive_parameter(self.opt_k_factor_air)

        kappa_phys_win_air = kappa_factor_win_air * self.initial_kappa_win_air
        tau_phys_ext_air = tau_factor_ext_air * self.initial_tau_ext_air
        k_phys_air = k_factor_air * self.initial_k_air

        theta_solar = self._positive_parameter(self.raw_theta_solar)
        alpha = keras.activations.sigmoid(self.opt_alpha)
        beta = keras.activations.sigmoid(self.opt_beta)

        v_flow_m3_s = v_flow_ahu / 3600.0
        h_ahu = self.rho_air * self.cp_air * v_flow_m3_s

        V_flow_w_h = self.V_flow_w_h_max * (keras.ops.exp(self.valve_a * y_valve_h / 100) - 1) / (keras.ops.exp(self.valve_a) - 1)
        QT = self.m_QT * v_flow_ahu + self.m_QT2 * v_flow_ahu**2
        KQ =  self.KQ_w3 * V_flow_w_h**3 + self.KQ_w2 * V_flow_w_h**2 + self.KQ_w1 * V_flow_w_h + self.KQ_w1a1 * v_flow_ahu * V_flow_w_h + self.KQ_w1a2 * v_flow_ahu**2 * V_flow_w_h + self.KQ_w2a1 * v_flow_ahu * V_flow_w_h**2
        q_did = QT * (t_sup_w_h - t_air) * KQ / (1 + QT * 0.86 / (2*V_flow_w_h + 1))

        q_solar = theta_solar * h_dir_nor

        if self.predict_delta:
            delta_t_air = y_air_pred / self.time_step
        else:
            delta_t_air = (y_air_pred - t_air) / self.time_step

        term_delta_t_air = tau_phys_ext_air * delta_t_air
        term_t_air = (1.0 + tau_phys_ext_air * kappa_phys_win_air + h_ahu * tau_phys_ext_air * k_phys_air) * t_air
        term_amb = tau_phys_ext_air * kappa_phys_win_air * t_amb
        term_ahu = h_ahu * tau_phys_ext_air * k_phys_air * t_ahu_sup
        term_did = tau_phys_ext_air * k_phys_air * q_did
        term_solar = alpha * tau_phys_ext_air * k_phys_air * q_solar

        if self.use_internal_gains:
            term_int = beta * tau_phys_ext_air * k_phys_air * q_int
        else:
            term_int = 0.0

        t_w = term_delta_t_air + term_t_air - term_amb - term_ahu - term_solar - term_int - term_did

        return t_w
    

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0][:-1]) + (1,)
    

    def get_config(self):
        """
        
        """
        config = super().get_config()

        config.update(
            {
                "time_step": self.time_step,
                "r_win": self.initial_r_win,
                "r_ext": self.initial_r_ext,
                "c_air": self.initial_c_air,
                "theta_solar_init": self.theta_solar_init,
                "alpha_init": self.alpha_init,
                "beta_init": self.beta_init,
                "t_air_index": self.t_air_index,
                "t_amb_index": self.t_amb_index,
                "v_flow_ahu_index": self.v_flow_ahu_index,
                "t_ahu_sup_index": self.t_ahu_sup_index,
                "t_sup_w_h_index": self.t_sup_w_h_index,
                "y_valve_h_index": self.y_valve_h_index,
                "h_dir_nor_index": self.h_dir_nor_index,
                "q_int_index": self.q_int_index,
                "predict_delta": self.predict_delta,
                "use_internal_gains": self.use_internal_gains,
                "trainable_rc": self.trainable_rc,
                "epsilon": self.epsilon,
            }
        )

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

@keras.saving.register_keras_serializable(package="custom_layer", name="RC2R2CGokhalePhysNetLayer",)
class RC2R2CGokhalePhysNetLayer(keras.Layer):
    """
    
    """
    def __init__(
        self,
        time_step: float,
        r_win: float,
        r_ext: float,
        c_air: float,
        t_air_index: int,
        t_amb_index: int,
        theta_solar_init: float = 1.75,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        v_flow_ahu_index: FeatureIndex = None,
        t_ahu_sup_index: FeatureIndex = None,
        t_sup_w_h_index: FeatureIndex = None,
        y_valve_h_index: FeatureIndex = None,
        h_dir_nor_index: FeatureIndex = None,
        q_int_index: FeatureIndex = None,
        predict_delta: bool = True,
        use_internal_gains: bool = False,
        trainable_rc: bool = False,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if time_step <= 0:
            raise ValueError("time_step must be positive")
        
        if r_win <= 0:
            raise ValueError("resistance of the window must be positive")
        
        if r_ext <= 0:
            raise ValueError("resistance of the wall must be positive")
        
        if c_air <= 0:
            raise ValueError("capacitance of the air must be positive")
        
        if theta_solar_init <= 0:
            raise ValueError("theta_solar_init must be positive")
        
        self.time_step = float(time_step)

        self.initial_r_win = float(r_win)
        self.initial_r_ext = float(r_ext)
        self.initial_c_air = float(c_air)
        self.theta_solar_init = float(theta_solar_init)
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)

        self.initial_tau_win_air = self.initial_r_win * self.initial_c_air
        self.initial_kappa_win_air = 1.0 / self.initial_tau_win_air
        self.initial_tau_ext_air = self.initial_r_ext * self.initial_c_air
        self.initial_k_air = 1.0 / self.initial_c_air

        self.t_air_index = int(t_air_index)
        self.t_amb_index = int(t_amb_index)

        self.v_flow_ahu_index = v_flow_ahu_index
        self.t_ahu_sup_index = t_ahu_sup_index
        self.t_sup_w_h_index = t_sup_w_h_index
        self.y_valve_h_index = y_valve_h_index
        self.h_dir_nor_index = h_dir_nor_index
        self.q_int_index = q_int_index

        self.predict_delta = bool(predict_delta)
        self.use_internal_gains = bool(use_internal_gains)
        self.trainable_rc = bool(trainable_rc)

        if self.use_internal_gains and not self.trainable_rc:
            raise ValueError("Configuration error: 'use_internal_gains' can only be True if 'trainable_rc' is also set to True.")
        
        self.epsilon = float(epsilon)

        self.rho_air = 1.204
        self.cp_air = 1005.0
        self.V_flow_w_h_max = 100
        self.valve_a = 3.2
        self.m_QT = 0.4464068811
        self.m_QT2 = -0.0003313083
        self.KQ_w1 = 0.0199019187
        self.KQ_w1a1 = -0.0000048959
        self.KQ_w2 = -0.0001255783
        self.KQ_w1a2 = -0.0000000062
        self.KQ_w2a1 = 0.0000000597
        self.KQ_w3 = 0.0000002721

    def build(self, input_shape):
        """
        
        """
        theta_solar_initializer = _inverse_softplus(self.theta_solar_init)
        initializer_val = _inverse_softplus(1.0)

        alpha_initializer = _inverse_sigmoid(self.alpha_init)
        beta_initializer = _inverse_sigmoid(self.beta_init)

        self.opt_kappa_factor_win_air = self.add_weight(
            name="opt_kappa_factor_win_air",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.opt_tau_factor_ext_air = self.add_weight(
            name="opt_tau_factor_ext_air",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc
        )

        self.opt_k_factor_air = self.add_weight(
            name="opt_k_factor_air",
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

        self.opt_alpha = self.add_weight(
            name="opt_alpha",
            shape=(1,),
            initializer=keras.initializers.Constant(alpha_initializer),
            trainable=True
        )

        self.opt_beta = self.add_weight(
            name="opt_beta",
            shape=(1,),
            initializer=keras.initializers.Constant(beta_initializer),
            trainable=self.use_internal_gains
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
    

    def _physical_calculation(self, x_eval, t_air_eval, delta_t_air_eval, x_did=None, t_air_did=None):
        """
        
        """
        if x_did is None:
            x_did = x_eval

        if t_air_did is None:
            t_air_did = t_air_eval

        t_amb = self._take_feature(
            inputs=x_eval,
            feature_index=self.t_amb_index,
            reference=t_air_eval,
        )

        v_flow_ahu = self._take_feature(
            inputs=x_eval, 
            feature_index=self.v_flow_ahu_index, 
            reference=t_air_eval
        )

        t_ahu_sup = self._take_feature(
            inputs=x_eval, 
            feature_index=self.t_ahu_sup_index, 
            reference=t_air_eval
        )

        t_sup_w_h = self._take_feature(
            inputs=x_did,
            feature_index=self.t_sup_w_h_index,
            reference=t_air_did
        )

        v_flow_ahu_did = self._take_feature(
            inputs=x_did,
            feature_index=self.v_flow_ahu_index,
            reference=t_air_did
        )

        y_valve_h = self._take_feature(
            inputs=x_did,
            feature_index=self.y_valve_h_index,
            reference=t_air_did
        )

        h_dir_nor = self._take_feature(
            inputs=x_eval, 
            feature_index=self.h_dir_nor_index, 
            reference=t_air_eval
        )

        
        q_int = self._take_feature(
            inputs=x_eval, 
            feature_index=self.q_int_index, 
            reference=t_air_eval
        )

        kappa_factor_win_air = self._positive_parameter(self.opt_kappa_factor_win_air)
        tau_factor_ext_air = self._positive_parameter(self.opt_tau_factor_ext_air)
        k_factor_air = self._positive_parameter(self.opt_k_factor_air)

        kappa_phys_win_air = kappa_factor_win_air * self.initial_kappa_win_air
        tau_phys_ext_air = tau_factor_ext_air * self.initial_tau_ext_air
        k_phys_air = k_factor_air * self.initial_k_air

        theta_solar = self._positive_parameter(self.raw_theta_solar)
        alpha = keras.activations.sigmoid(self.opt_alpha)
        beta = keras.activations.sigmoid(self.opt_beta)

        v_flow_m3_s = v_flow_ahu / 3600.0
        h_ahu = self.rho_air * self.cp_air * v_flow_m3_s

        V_flow_w_h = self.V_flow_w_h_max * (keras.ops.exp(self.valve_a * y_valve_h / 100) - 1) / (keras.ops.exp(self.valve_a) - 1)
        QT = self.m_QT * v_flow_ahu_did + self.m_QT2 * v_flow_ahu_did**2
        KQ =  self.KQ_w3 * V_flow_w_h**3 + self.KQ_w2 * V_flow_w_h**2 + self.KQ_w1 * V_flow_w_h + self.KQ_w1a1 * v_flow_ahu_did * V_flow_w_h + self.KQ_w1a2 * v_flow_ahu_did**2 * V_flow_w_h + self.KQ_w2a1 * v_flow_ahu_did * V_flow_w_h**2
        q_did = QT * (t_sup_w_h - t_air_did) * KQ / (1 + QT * 0.86 / (2*V_flow_w_h + 1))

        q_solar = theta_solar * h_dir_nor

        term_delta_t_air = tau_phys_ext_air * delta_t_air_eval
        term_t_air = (1.0 + tau_phys_ext_air * kappa_phys_win_air + h_ahu * tau_phys_ext_air * k_phys_air) * t_air_eval
        term_amb = tau_phys_ext_air * kappa_phys_win_air * t_amb
        term_ahu = h_ahu * tau_phys_ext_air * k_phys_air * t_ahu_sup
        term_did = tau_phys_ext_air * k_phys_air * q_did
        term_solar = alpha * tau_phys_ext_air * k_phys_air * q_solar

        if self.use_internal_gains:
            term_int = beta * tau_phys_ext_air * k_phys_air * q_int
        else:
            term_int = 0.0

        t_w_phys = term_delta_t_air + term_t_air - term_amb - term_ahu - term_solar - term_int - term_did

        return t_w_phys
    

    def call(self, inputs, **kwargs):
        if not isinstance(inputs,(list, tuple)) or len(inputs) != 4:
            raise ValueError("RC2R2CGokhalePhysNetLayer expects four inputs: [x_k, y_pred_k, x_k1, y_true_k1]!")
        
        x_k, y_pred_k, x_k1, y_true_k1 = inputs

        t_air_k = self._take_feature(
            inputs=x_k,
            feature_index=self.t_air_index,
            reference=x_k
        )

        if self.predict_delta:
            t_air_k1_pred = t_air_k + y_pred_k
        else:
            t_air_k1_pred = y_pred_k

        t_air_k1 = self._take_feature(
            inputs=x_k1,
            feature_index=self.t_air_index,
            reference=t_air_k
        )

        if self.predict_delta:
            t_air_k2 = t_air_k1 + y_true_k1
        else:
            t_air_k2 = y_true_k1

        delta_t_air_k1 =(t_air_k2 - t_air_k) / (self.time_step * 2.0)

        return self._physical_calculation(x_eval=x_k1, t_air_eval=t_air_k1_pred, delta_t_air_eval=delta_t_air_k1, x_did=x_k, t_air_did=t_air_k)
    

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0][:-1]) + (1,)
    

    def get_config(self):
        """
        
        """
        config = super().get_config()

        config.update(
            {
                "time_step": self.time_step,
                "r_win": self.initial_r_win,
                "r_ext": self.initial_r_ext,
                "c_air": self.initial_c_air,
                "theta_solar_init": self.theta_solar_init,
                "alpha_init": self.alpha_init,
                "beta_init": self.beta_init,
                "t_air_index": self.t_air_index,
                "t_amb_index": self.t_amb_index,
                "v_flow_ahu_index": self.v_flow_ahu_index,
                "t_ahu_sup_index": self.t_ahu_sup_index,
                "t_sup_w_h_index": self.t_sup_w_h_index,
                "y_valve_h_index": self.y_valve_h_index,
                "h_dir_nor_index": self.h_dir_nor_index,
                "q_int_index": self.q_int_index,
                "predict_delta": self.predict_delta,
                "use_internal_gains": self.use_internal_gains,
                "trainable_rc": self.trainable_rc,
                "epsilon": self.epsilon,
            }
        )

        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

@keras.saving.register_keras_serializable(package="custom_layer", name="RC2R2CGokhalePhysNetWallDynamicsLayer",)
class RC2R2CGokhalePhysNetWallDynamicsLayer(RC2R2CGokhalePhysNetLayer):
    """
    
    """
    def __init__(
            self,
            r_ext_rem: float,
            c_ext: float,
            **kwargs
    ):
        
        super().__init__(**kwargs)

        if r_ext_rem <= 0:
            raise ValueError("remaining external resistance of the wall must be positive")

        if c_ext <= 0:
            raise ValueError("capacitance of the wall must be positive")
        
        self.initial_r_ext_rem = float(r_ext_rem)
        self.initial_c_ext = float(c_ext)

        self.initial_tau_ext_rem_wall = self.initial_r_ext_rem * self.initial_c_ext
        self.initial_kappa_ext_rem_wall = 1.0 / self.initial_tau_ext_rem_wall
        self.initial_k_wall = 1.0 / self.initial_c_ext

    def build(self, input_shape):
        super().build(input_shape)

        initializer_val = _inverse_softplus(1.0)

        self.opt_kappa_factor_ext_rem_wall = self.add_weight(
            name="opt_kappa_factor_ext_rem_wall",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc,
        )

        self.opt_k_factor_wall = self.add_weight(
            name="opt_k_factor_wall",
            shape=(1,),
            initializer=keras.initializers.Constant(initializer_val),
            trainable=self.trainable_rc,
        )

    def wall_dynamics(self, inputs, **kwargs):
        """
        
        """
        if not isinstance(inputs,(list, tuple)) or len(inputs) != 2:
            raise ValueError("RC2R2CGokhalePhysNetWallDynamicsLayer expects two inputs: [x_k, t_wall_k]!")
        
        x_k, t_wall_k = inputs

        t_air_k = self._take_feature(
            inputs=x_k,
            feature_index=self.t_air_index,
            reference=x_k
        )

        t_amb = self._take_feature(
            inputs=x_k,
            feature_index=self.t_amb_index,
            reference=t_air_k,
        )

        h_dir_nor = self._take_feature(
            inputs=x_k, 
            feature_index=self.h_dir_nor_index, 
            reference=t_air_k
        )

        
        q_int = self._take_feature(
            inputs=x_k, 
            feature_index=self.q_int_index, 
            reference=t_air_k
        )

        tau_factor_ext_air = self._positive_parameter(self.opt_tau_factor_ext_air)
        k_factor_air = self._positive_parameter(self.opt_k_factor_air)
        kappa_factor_ext_rem_wall = self._positive_parameter(self.opt_kappa_factor_ext_rem_wall)
        k_factor_wall = self._positive_parameter(self.opt_k_factor_wall)

        tau_phys_ext_air = tau_factor_ext_air * self.initial_tau_ext_air
        k_phys_air = k_factor_air * self.initial_k_air
        kappa_phys_ext_rem_wall = kappa_factor_ext_rem_wall * self.initial_kappa_ext_rem_wall
        k_phys_wall = k_factor_wall * self.initial_k_wall
        r_phys_wall_air = tau_phys_ext_air * k_phys_air
        kappa_phys_ext_wall = k_phys_wall / r_phys_wall_air

        theta_solar = self._positive_parameter(self.raw_theta_solar)
        alpha = keras.activations.sigmoid(self.opt_alpha)
        beta = keras.activations.sigmoid(self.opt_beta)

        term_solar = (1.0 - alpha) * k_phys_wall * theta_solar * h_dir_nor
        term_air_to_wall = kappa_phys_ext_wall * (t_air_k - t_wall_k)
        term_amb_to_wall = kappa_phys_ext_rem_wall * (t_amb - t_wall_k)

        delta_t_wall = term_solar + term_air_to_wall + term_amb_to_wall

        if self.use_internal_gains:
            delta_t_wall = delta_t_wall + (1.0 - beta) * k_phys_wall * q_int

        t_wall_k1_phys = t_wall_k + self.time_step * delta_t_wall

        return t_wall_k1_phys
    
    def get_config(self):
        """
        
        """
        config = super().get_config()

        config.update(
            {
                "r_ext_rem": self.initial_r_ext_rem,
                "c_ext": self.initial_c_ext,
            }
        )

        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config) 