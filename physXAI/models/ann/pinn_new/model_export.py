import keras
import numpy as np

core_model_name = 'core_model'

def get_core_model(training_model: keras.Model) -> keras.Model:
    """
    
    """
    if not isinstance(training_model, keras.Model):
        raise TypeError("training_model must be a trained keras.Model!")

    core_model = getattr(training_model, core_model_name, None)

    if core_model is None:
        try:
            core_model = training_model.get_layer(core_model_name)
        except ValueError as error:
            raise ValueError(f"the model '{training_model.name}' does not contain a submodel named '{core_model_name}'!") from error

    if not isinstance(core_model, keras.Model):
        raise TypeError(f"'{core_model_name}' was found but is not a keras.Model!")

    return core_model


def build_mpc_model(training_model: keras.Model) -> keras.Model:
    """
    
    """
    core_model = get_core_model(training_model)

    if len(core_model.inputs) != 1:
        raise ValueError(f"'{core_model_name}' must have exactly one input, but has {len(core_model.inputs)} inputs!")

    if not core_model.outputs:
        raise ValueError(f"'{core_model_name}' does not have an output!")

    change_t_air = core_model.outputs[0]

    mpc_model = keras.Model(
        inputs=core_model.inputs[0],
        outputs=change_t_air,
        name="pinn_mpc_model",
    )

    return mpc_model


def export_model_for_mpc(training_model: keras.Model, validation_inputs: np.ndarray, save_path: str) -> str:
    """
    
    """
    mpc_model = build_mpc_model(training_model)

    if len(mpc_model.inputs) != 1:
        raise ValueError("The MPC model must have exactly one input tensor!")

    if len(mpc_model.outputs) != 1:
        raise ValueError("The MPC model must have exactly one output tensor!")

    validation_inputs = validation_inputs[:256]

    prediction_before_saving = mpc_model.predict(validation_inputs, verbose=0)

    if not save_path.endswith(".keras"):
        save_path += ".keras"

    mpc_model.save(save_path)

    loaded_model = keras.saving.load_model(save_path)

    prediction_after_loading = loaded_model.predict(validation_inputs, verbose=0)

    np.testing.assert_allclose(
        prediction_before_saving,
        prediction_after_loading,
        rtol=1e-5,
        atol=1e-6,
    )

    return save_path

def get_validation_inputs(td) -> np.ndarray:
    """
    
    """
    for inputs in (
        td.X_test_single,
        td.X_val_single,
        td.X_train_single,
    ):
        if inputs is not None and len(inputs) > 0:
            return inputs

    raise ValueError("No data are avaiable for checking the exported model!")