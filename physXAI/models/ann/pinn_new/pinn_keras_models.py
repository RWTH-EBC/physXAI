import keras
import tensorflow as tf

from physXAI.models.ann.pinn_new.rc_layers import RC1R1CLayer, RC2R2CPhysNetLayer, RC2R2CGokhalePhysNetLayer, RC2R2CGokhalePhysNetWallDynamicsLayer


@keras.saving.register_keras_serializable(package="custom_model", name="RC1R1CKerasModel")
class RC1R1CKerasModel(keras.Model):
    """
    
    """

    def __init__(
            self,
            inputs,
            outputs,
            core_model: keras.Model,
            physics_layer: RC1R1CLayer,
            physics_loss_weight: float = 1.0,
            predict_delta: bool = True,
            t_air_index : int = 0,
            **kwargs,
    ):
        super().__init__(
            inputs=inputs, 
            outputs=outputs, 
            **kwargs,
        )

        self.core_model = core_model
        self.physics_layer = physics_layer

        #if not(layer is physics_layer for layer in self._layers):
        self._layers.append(physics_layer)

        self.physics_loss_weight = float(physics_loss_weight)
        self.predict_delta = predict_delta
        self.t_air_index = t_air_index

        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.prediction_loss_tracker = keras.metrics.Mean(name='prediction_loss')
        self.physics_loss_tracker = keras.metrics.Mean(name='physics_loss')
        self.rmse_tracker = keras.metrics.RootMeanSquaredError(name='rmse')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.prediction_loss_tracker, self.physics_loss_tracker, self.rmse_tracker]

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "core_model": keras.saving.serialize_keras_object(self.core_model),
                "physics_layer": keras.saving.serialize_keras_object(self.physics_layer),
                "physics_loss_weight": self.physics_loss_weight,
                "predict_delta": self.predict_delta,
                "t_air_index": self.t_air_index,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        core_model = keras.saving.deserialize_keras_object(config.pop("core_model"))
        physics_layer = keras.saving.deserialize_keras_object(config.pop("physics_layer"))
        physics_loss_weight = config.pop("physics_loss_weight")
        predict_delta = config.pop("predict_delta")
        t_air_index = config.pop("t_air_index")
        core_input = core_model.inputs[0]
        pinn_input = keras.layers.Input(
            shape=tuple(core_input.shape[1:]),
            dtype=core_input.dtype,
            name='pinn_input',
        )

        y_nn = core_model(pinn_input)

        return cls(
            inputs=pinn_input,
            outputs=y_nn,
            core_model=core_model,
            physics_layer=physics_layer,
            physics_loss_weight=physics_loss_weight,
            predict_delta=predict_delta,
            t_air_index=t_air_index,
            **config,
        )

    def train_step(self, data):
        inputs, targets = data

        model_dtype = self.core_model.compute_dtype
        inputs = keras.ops.cast(inputs, model_dtype)
        targets = keras.ops.cast(targets, model_dtype)

        with tf.GradientTape() as tape:
            predictions = self.core_model(inputs, training=True)

            physics_predictions = self.physics_layer(inputs, training=True)

            if not self.predict_delta:
                t_air = inputs[:, self.t_air_index:(self.t_air_index + 1)]

                physics_predictions = t_air + physics_predictions

            prediction_loss = keras.ops.mean(keras.ops.square(targets - predictions))

            physics_loss = keras.ops.mean(keras.ops.square(predictions - physics_predictions))

            total_loss = prediction_loss + self.physics_loss_weight * physics_loss

            if self.losses:
                total_loss = total_loss + tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        batch_weight = keras.ops.cast(keras.ops.shape(targets)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
        self.rmse_tracker.update_state(targets, predictions)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
                }

    def test_step(self, data):
        inputs, targets = data

        model_dtype = self.core_model.compute_dtype
        inputs = keras.ops.cast(inputs, model_dtype)
        targets = keras.ops.cast(targets, model_dtype)

        predictions = self.core_model(inputs, training=False)

        physics_predictions = self.physics_layer(inputs, training=False)

        if not self.predict_delta:
            t_air = inputs[:, self.t_air_index:(self.t_air_index + 1)]

            physics_predictions = t_air + physics_predictions

        prediction_loss = keras.ops.mean(keras.ops.square(targets - predictions))

        physics_loss = keras.ops.mean(keras.ops.square(predictions - physics_predictions))

        total_loss = prediction_loss + self.physics_loss_weight * physics_loss

        if self.losses:
            total_loss = total_loss + tf.add_n(self.losses)

        batch_weight = keras.ops.cast(keras.ops.shape(targets)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
        self.rmse_tracker.update_state(targets, predictions)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
                }


@keras.saving.register_keras_serializable(package="custom_model", name="RC2R2CPhysNetKerasModel")
class RC2R2CPhysNetKerasModel(keras.Model):
    """
    
    """

    def __init__(
            self,
            inputs,
            outputs,
            core_model: keras.Model,
            physics_layer: RC2R2CPhysNetLayer,
            physics_loss_weight: float = 1.0,
            prediction_loss_scale: float = 1.0,
            physics_loss_scale: float = 1.0,
            **kwargs,
    ):
        super().__init__(
            inputs=inputs, 
            outputs=outputs, 
            **kwargs,
        )

        self.core_model = core_model
        self.physics_layer = physics_layer

        self._layers.append(physics_layer)

        self.physics_loss_weight = float(physics_loss_weight)

        self.prediction_loss_scale = float(prediction_loss_scale)
        self.physics_loss_scale = float(physics_loss_scale)

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
                "prediction_loss_scale": self.prediction_loss_scale,
                "physics_loss_scale": self.physics_loss_scale,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        core_model = keras.saving.deserialize_keras_object(config.pop("core_model"))
        physics_layer = keras.saving.deserialize_keras_object(config.pop("physics_layer"))
        physics_loss_weight = config.pop("physics_loss_weight")
        prediction_loss_scale = config.pop("prediction_loss_scale")
        physics_loss_scale = config.pop("physics_loss_scale")
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
            prediction_loss_scale=prediction_loss_scale,
            physics_loss_scale=physics_loss_scale,
            **config,
        )

    def train_step(self, data):
        inputs, targets = data

        model_dtype = self.core_model.compute_dtype
        inputs = keras.ops.cast(inputs, model_dtype)
        targets = keras.ops.cast(targets, model_dtype)

        with tf.GradientTape() as tape:
            predictions, z_latent = self.core_model(inputs, training=True)

            z_physic = self.physics_layer([inputs, predictions], training=True)

            prediction_loss = keras.ops.mean(keras.ops.square((targets - predictions) / self.prediction_loss_scale))

            physics_loss = keras.ops.mean(keras.ops.square((z_latent - z_physic) / self.physics_loss_scale))

            total_loss = prediction_loss + self.physics_loss_weight * physics_loss

            if self.losses:
                total_loss = total_loss + tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        batch_weight = keras.ops.cast(keras.ops.shape(targets)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
        self.rmse_tracker.update_state(targets, predictions)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }

    def test_step(self, data):
        inputs, targets = data

        model_dtype = self.core_model.compute_dtype
        inputs = keras.ops.cast(inputs, model_dtype)
        targets = keras.ops.cast(targets, model_dtype)

        predictions, z_latent = self.core_model(inputs, training=False)

        z_physic = self.physics_layer([inputs, predictions], training=False)

        prediction_loss = keras.ops.mean(keras.ops.square((targets - predictions) / self.prediction_loss_scale))

        physics_loss = keras.ops.mean(keras.ops.square((z_latent - z_physic) / self.physics_loss_scale))

        total_loss = prediction_loss + self.physics_loss_weight * physics_loss

        if self.losses:
            total_loss = total_loss + tf.add_n(self.losses)

        batch_weight = keras.ops.cast(keras.ops.shape(targets)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
        self.rmse_tracker.update_state(targets, predictions)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }

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
            prediction_loss_scale: float = 1.0,
            physics_loss_scale: float = 1.0,
            **kwargs,
    ):
        super().__init__(
            inputs=inputs, 
            outputs=outputs, 
            **kwargs,
        )

        self.core_model = core_model
        self.physics_layer = physics_layer

        #if not(layer is physics_layer for layer in self._layers):
        self._layers.append(physics_layer)

        self.physics_loss_weight = float(physics_loss_weight)

        self.prediction_loss_scale = float(prediction_loss_scale)
        self.physics_loss_scale = float(physics_loss_scale)

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
                "prediction_loss_scale": self.prediction_loss_scale,
                "physics_loss_scale": self.physics_loss_scale,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        core_model = keras.saving.deserialize_keras_object(config.pop("core_model"))
        physics_layer = keras.saving.deserialize_keras_object(config.pop("physics_layer"))
        physics_loss_weight = config.pop("physics_loss_weight")
        prediction_loss_scale = config.pop("prediction_loss_scale")
        physics_loss_scale = config.pop("physics_loss_scale")
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
            prediction_loss_scale=prediction_loss_scale,
            physics_loss_scale=physics_loss_scale,
            **config,
        )
    
    def train_step(self, data):
        (x_k, x_k1), y_k1 = data

        model_dtype = self.core_model.compute_dtype
        x_k = keras.ops.cast(x_k, model_dtype)
        x_k1 = keras.ops.cast(x_k1, model_dtype)
        y_k1 = keras.ops.cast(y_k1, model_dtype)

        with tf.GradientTape() as tape:
            y_pred_k, _ = self.core_model(x_k, training=True)

            y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=True)

            z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=True)

            prediction_loss = keras.ops.mean(keras.ops.square((y_k1 - y_pred_k1) / self.prediction_loss_scale))

            physics_loss = keras.ops.mean(keras.ops.square((z_latent_k1 - z_phys_k1) / self.physics_loss_scale))

            total_loss = prediction_loss + self.physics_loss_weight * physics_loss

            if self.losses:
                total_loss = total_loss + tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        batch_weight = keras.ops.cast(keras.ops.shape(y_k1)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
        self.rmse_tracker.update_state(y_k1, y_pred_k1)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }
    
    def test_step(self, data):
        (x_k, x_k1), y_k1 = data

        model_dtype = self.core_model.compute_dtype
        x_k = keras.ops.cast(x_k, model_dtype)
        x_k1 = keras.ops.cast(x_k1, model_dtype)
        y_k1 = keras.ops.cast(y_k1, model_dtype)

        y_pred_k, _ = self.core_model(x_k, training=False)

        y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=False)

        z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=False)

        prediction_loss = keras.ops.mean(keras.ops.square((y_k1 - y_pred_k1) / self.prediction_loss_scale))
        
        physics_loss = keras.ops.mean(keras.ops.square((z_latent_k1 - z_phys_k1) / self.physics_loss_scale))
        
        total_loss = prediction_loss + self.physics_loss_weight * physics_loss

        if self.losses:
            total_loss = total_loss + tf.add_n(self.losses)

        batch_weight = keras.ops.cast(keras.ops.shape(y_k1)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
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
            prediction_loss_scale: float = 1.0,
            physics_loss_scale: float = 1.0,
            **kwargs,
    ):
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            core_model=core_model,
            physics_layer=physics_layer,
            physics_loss_weight=physics_loss_weight,
            prediction_loss_scale=prediction_loss_scale,
            physics_loss_scale=physics_loss_scale,
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
        prediction_loss_scale = config.pop("prediction_loss_scale")
        physics_loss_scale = config.pop("physics_loss_scale")
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
            prediction_loss_scale=prediction_loss_scale,
            physics_loss_scale=physics_loss_scale,
            **config,
        )
   
    def train_step(self, data):
        (x_k, x_k1), y_k1 = data

        model_dtype = self.core_model.compute_dtype
        x_k = keras.ops.cast(x_k, model_dtype)
        x_k1 = keras.ops.cast(x_k1, model_dtype)
        y_k1 = keras.ops.cast(y_k1, model_dtype)

        with tf.GradientTape() as tape:
            y_pred_k, z_latent_k = self.core_model(x_k, training=True)
            y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=True)

            z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=True)

            z_dyn_phys_k1 = self.physics_layer.wall_dynamics([x_k, z_latent_k],training=True)

            prediction_loss = keras.ops.mean(keras.ops.square((y_k1 - y_pred_k1) / self.prediction_loss_scale))

            physics_loss = keras.ops.mean(keras.ops.square((z_latent_k1 - z_phys_k1) / self.physics_loss_scale))

            wall_dynamics_loss = keras.ops.mean(keras.ops.square((z_latent_k1 - z_dyn_phys_k1) / self.physics_loss_scale))

            total_loss = prediction_loss + self.physics_loss_weight * physics_loss + self.wall_dynamics_loss_weight * wall_dynamics_loss

            if self.losses:
                total_loss = total_loss + tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        batch_weight = keras.ops.cast(keras.ops.shape(y_k1)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
        self.wall_dynamics_loss_tracker.update_state(wall_dynamics_loss, sample_weight=batch_weight)
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

        model_dtype = self.core_model.compute_dtype
        x_k = keras.ops.cast(x_k, model_dtype)
        x_k1 = keras.ops.cast(x_k1, model_dtype)
        y_k1 = keras.ops.cast(y_k1, model_dtype)

        y_pred_k, z_latent_k = self.core_model(x_k, training=False)
        y_pred_k1, z_latent_k1 = self.core_model(x_k1, training=False)

        z_phys_k1 = self.physics_layer([x_k, y_pred_k, x_k1, y_k1], training=False)

        z_dyn_phys_k1 = self.physics_layer.wall_dynamics([x_k, z_latent_k],training=False)

        prediction_loss = keras.ops.mean(keras.ops.square((y_k1 - y_pred_k1) / self.prediction_loss_scale))
        
        physics_loss = keras.ops.mean(keras.ops.square((z_latent_k1 - z_phys_k1) / self.physics_loss_scale))

        wall_dynamics_loss = keras.ops.mean(keras.ops.square((z_latent_k1 - z_dyn_phys_k1) / self.physics_loss_scale))

        total_loss = prediction_loss + self.physics_loss_weight * physics_loss + self.wall_dynamics_loss_weight * wall_dynamics_loss

        if self.losses:
            total_loss = total_loss + tf.add_n(self.losses)

        batch_weight = keras.ops.cast(keras.ops.shape(y_k1)[0], total_loss.dtype)

        self.total_loss_tracker.update_state(total_loss, sample_weight=batch_weight)
        self.prediction_loss_tracker.update_state(prediction_loss, sample_weight=batch_weight)
        self.physics_loss_tracker.update_state(physics_loss, sample_weight=batch_weight)
        self.wall_dynamics_loss_tracker.update_state(wall_dynamics_loss, sample_weight=batch_weight)
        self.rmse_tracker.update_state(y_k1, y_pred_k1)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
            "wall_dynamics_loss": self.wall_dynamics_loss_tracker.result(),
            "rmse": self.rmse_tracker.result(),
        }