import keras
import tensorflow as tf

from physXAI.models.ann.pinn_new.rc_layers import RC2R2CGokhalePhysNetLayer, RC2R2CGokhalePhysNetWallDynamicsLayer



@keras.saving.register_keras_serializable(package="custom_model", name="RC2R2CGokhalePhysNetKerasModel")
class RC2R2CGokhalePhysNetKerasModel(keras.Model):
    """
    
    """

    def __init__(
            self,
            inputs,
            outputs,
            core_model: keras.Model,
            physics_layer: RC2R2CGokhalePhysNetLayer,
            physics_loss_weight: float = 1.0,
            **kwargs,
    ):
        super().__init__(
            inputs=inputs, 
            outputs=outputs, 
            **kwargs,
        )

        self.core_model = core_model
        self.physics_layer = physics_layer
        self.physics_loss_weight = float(physics_loss_weight)

        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.prediction_loss_tracker = keras.metrics.Mean(name='prediction_loss')
        self.physics_loss_tracker = keras.metrics.Mean(name='physics_loss')
        self.rmse_tracker = keras.metrics.RootMeanSquaredError(name='rmse')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.prediction_loss_tracker, self.physics_loss_tracker, self.rmse_tracker]
    
    def predict_with_latent(self, inputs, training=False):
        return self.core_model(inputs, training=training)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "core_model": keras.saving.serialize_keras_object(self.core_model),
                "physics_layer": keras.saving.serialize_keras_object(self.physics_layer),
                "physics_loss_weight": self.physics_loss_weight,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        core_model = keras.saving.deserialize_keras_object(config.pop("core_model"))
        physics_layer = keras.saving.deserialize_keras_object(config.pop("physics_layer"))
        physics_loss_weight = config.pop("physics_loss_weight")
        core_input = core_model.inputs[0]
        pinn_input = keras.layers.Input(
            shape=tuple(core_input.shape[1:]),
            dtype=core_input.dtype,
            name='pinn_input',
        )

        y_nn, _ = core_model(pinn_input)

        return cls(
            inputs=pinn_input,
            outputs=y_nn,
            core_model=core_model,
            physics_layer=physics_layer,
            physics_loss_weight=physics_loss_weight,
            **config,
        )
    
    def train_step(self, data):
        (x_k, x_k1), y_k1 = data

        with tf.GradientTape() as tape:
            y_pred_k, _ = self.core_model(x_k, training=True)

            y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=True)

            z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=True)

            prediction_loss = keras.ops.mean(keras.ops.square(y_k1 - y_pred_k1))

            physics_loss = keras.ops.mean(keras.ops.square(z_latent_k1 - z_phys_k1))

            total_loss = prediction_loss + self.physics_loss_weight * physics_loss

            if self.losses:
                total_loss = total_loss + tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.rmse_tracker.update_state(y_k1, y_pred_k1)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }
    
    def test_step(self, data):
        (x_k, x_k1), y_k1 = data

        y_pred_k, _ = self.core_model(x_k, training=False)

        y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=False)

        z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=False)

        prediction_loss = keras.ops.mean(keras.ops.square(y_k1 - y_pred_k1))

        physics_loss = keras.ops.mean(keras.ops.square(z_latent_k1 - z_phys_k1))

        total_loss = prediction_loss + self.physics_loss_weight * physics_loss

        if self.losses:
            total_loss = total_loss + tf.add_n(self.losses)

        self.total_loss_tracker.update_state(total_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.rmse_tracker.update_state(y_k1, y_pred_k1)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }


@keras.saving.register_keras_serializable(package="custom_model", name="RC2R2CGokhalePhysNetWallDynamicsKerasModel")
class RC2R2CGokhalePhysNetWallDynamicsKerasModel(RC2R2CGokhalePhysNetKerasModel):
    """
    
    """
    def __init__(
            self,
            inputs,
            outputs,
            core_model: keras.Model,
            physics_layer: RC2R2CGokhalePhysNetWallDynamicsLayer,
            physics_loss_weight: float = 1.0,
            wall_dynamics_loss_weight: float = 1.0,
            **kwargs,
    ):
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            core_model=core_model,
            physics_layer=physics_layer,
            physics_loss_weight=physics_loss_weight,
            **kwargs,
        )

        self.wall_dynamics_loss_weight = float(wall_dynamics_loss_weight)

        self.wall_dynamics_loss_tracker = keras.metrics.Mean(name="wall_dynamics_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.prediction_loss_tracker, self.physics_loss_tracker, self.wall_dynamics_loss_tracker, self.rmse_tracker]
    
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "wall_dynamics_loss_weight": self.wall_dynamics_loss_weight,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        core_model = keras.saving.deserialize_keras_object(config.pop("core_model"))
        physics_layer = keras.saving.deserialize_keras_object(config.pop("physics_layer"))
        physics_loss_weight = config.pop("physics_loss_weight")
        wall_dynamics_loss_weight = config.pop("wall_dynamics_loss_weight")
        core_input = core_model.inputs[0]
        pinn_input = keras.layers.Input(
            shape=tuple(core_input.shape[1:]),
            dtype=core_input.dtype,
            name='pinn_input',
        )

        y_nn, _ = core_model(pinn_input)

        return cls(
            inputs=pinn_input,
            outputs=y_nn,
            core_model=core_model,
            physics_layer=physics_layer,
            physics_loss_weight=physics_loss_weight,
            wall_dynamics_loss_weight=wall_dynamics_loss_weight,
            **config,
        )
   
    def train_step(self, data):
        (x_k, x_k1), y_k1 = data

        with tf.GradientTape() as tape:
            y_pred_k, z_latent_k = self.core_model(x_k, training=True)
            y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=True)

            z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=True)

            z_dyn_phys_k1 = self.physics_layer.wall_dynamics([x_k, z_latent_k],training=True)

            prediction_loss = keras.ops.mean(keras.ops.square(y_k1 - y_pred_k1))

            physics_loss = keras.ops.mean(keras.ops.square(z_latent_k1 - z_phys_k1))

            wall_dynamics_loss = keras.ops.mean(keras.ops.square(z_latent_k1 - z_dyn_phys_k1))

            total_loss = prediction_loss + self.physics_loss_weight * physics_loss + self.wall_dynamics_loss_weight * wall_dynamics_loss

            if self.losses:
                total_loss = total_loss + tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.wall_dynamics_loss_tracker.update_state(wall_dynamics_loss)
        self.rmse_tracker.update_state(y_k1, y_pred_k1)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "wall_dynamics_loss": self.wall_dynamics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }
    
    def test_step(self, data):
        (x_k, x_k1), y_k1 = data

        y_pred_k, z_latent_k = self.core_model(x_k, training=False)
        y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=False)

        z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=False)

        z_dyn_phys_k1 = self.physics_layer.wall_dynamics([x_k, z_latent_k],training=False)

        prediction_loss = keras.ops.mean(keras.ops.square(y_k1 - y_pred_k1))

        physics_loss = keras.ops.mean(keras.ops.square(z_latent_k1 - z_phys_k1))

        wall_dynamics_loss = keras.ops.mean(keras.ops.square(z_latent_k1 - z_dyn_phys_k1))

        total_loss = prediction_loss + self.physics_loss_weight * physics_loss + self.wall_dynamics_loss_weight * wall_dynamics_loss

        if self.losses:
            total_loss = total_loss + tf.add_n(self.losses)

        self.total_loss_tracker.update_state(total_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.wall_dynamics_loss_tracker.update_state(wall_dynamics_loss)
        self.rmse_tracker.update_state(y_k1, y_pred_k1)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "wall_dynamics_loss": self.wall_dynamics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }