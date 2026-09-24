import os
import pathlib
from unittest.mock import patch
import keras
import numpy as np
import pytest
######################################################################################################################
from physXAI.utils.logging import Logger
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep
from physXAI.preprocessing.constructed import Feature
from physXAI.models.ann.ann_design import RNNModel
from physXAI.models.modular.modular_ann import ModularNormalization
from physXAI.evaluation.closed_loop import closed_loop_predictions, derive_feedback


base_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'stored_data')


@pytest.fixture(autouse=True)
def disable_plotly_show():
    """Automatically disable plotly show() for all tests"""
    with patch('plotly.graph_objects.Figure.show'):
        yield


@pytest.fixture(scope='module')
def file_path():
    return os.path.join(pathlib.Path(__file__).resolve().parent.parent,
                        "data/bestest_hydronic_heat_pump/pid_data.csv")


@pytest.fixture
def recursive_data(file_path):
    """Multi step data of a model that predicts one of its own inputs."""
    Logger.setup_logger(base_path=base_path, folder_name='unittests\\test_closed_loop', override=True)
    inputs = ['weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u', 'reaTZon_y']
    prep = PreprocessingMultiStep(inputs=inputs, output='reaTZon_y', label_width=4, warmup_width=2)
    return prep.pipeline(file_path)


def test_derive_feedback(recursive_data):
    # the output is an input of the model, so it is fed back
    assert derive_feedback(recursive_data) == {'reaTZon_y': 'reaTZon_y'}

    # an output holding a difference is recognized by its name
    recursive_data.output = ['Change(reaTZon_y)']
    assert derive_feedback(recursive_data) == {'reaTZon_y': 'Change(reaTZon_y)'}

    # a model that does not predict any of its own inputs has no feedback
    recursive_data.output = ['something_else']
    assert derive_feedback(recursive_data) == dict()


def test_closed_loop_predictions(recursive_data):
    """Feeding the predictions back has to change everything but the first step."""
    m = RNNModel(epochs=1, rnn_layer='GRU')
    model = m.pipeline(recursive_data, save_model=False, plot=False)

    teacher_forced = recursive_data.y_train_pred
    closed_loop = recursive_data.y_train_pred_closed_loop
    assert closed_loop is not None, "the closed loop evaluation did not run"
    assert closed_loop.shape == teacher_forced.shape

    # the first step still uses the measurement, just like an MPC does
    assert np.allclose(closed_loop[:, 0, :], teacher_forced[:, 0, :], atol=1e-4)
    assert not np.allclose(closed_loop, teacher_forced, atol=1e-4)

    # the metrics are stored next to the teacher forced ones
    assert recursive_data.closed_loop_metrics is not None
    assert 'RMSE Test' in recursive_data.closed_loop_metrics.test_kpis
    assert len(recursive_data.closed_loop_metrics.rmse_test_l) == recursive_data.y_test.shape[1]
    assert recursive_data.get_config()['closed_loop_metrics'] is not None

    # starting from zero states, as an MPC does before it warms them up, is possible too
    from_zeros = closed_loop_predictions(model, recursive_data, warmup='zeros')
    assert from_zeros[0].shape == teacher_forced.shape


def test_closed_loop_is_skipped_without_feedback(file_path):
    """A model that does not predict any of its own inputs has nothing to feed back."""
    Logger.setup_logger(base_path=base_path, folder_name='unittests\\test_closed_loop', override=True)
    prep = PreprocessingMultiStep(inputs=['weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u'],
                                  output='reaTZon_y', label_width=4, warmup_width=2)
    td = prep.pipeline(file_path)

    m = RNNModel(epochs=1, rnn_layer='GRU')
    m.pipeline(td, save_model=False, plot=False)

    assert td.closed_loop_metrics is None
    assert td.metrics is not None


def test_feature_architecture(recursive_data):
    """A modular expression can build the inputs of the recurrent layer.

    This makes a physically motivated combination of features part of the model instead
    of the data, which an MPC would otherwise need a measured history of.
    """
    heat_pump = Feature('oveHeaPumY_u')
    ambient = Feature('weaSta_reaWeaTDryBul_y')
    zone = Feature('reaTZon_y')

    # a product of raw features, normalized afterwards
    heat_flow = heat_pump.input(normalize=False) * (zone.input(normalize=False)
                                                    - ambient.input(normalize=False))
    heat_flow.rename('heat_flow')
    architecture = [Feature(name).input() for name in recursive_data.columns]
    architecture.append(ModularNormalization(heat_flow))

    m = RNNModel(epochs=1, rnn_layer='GRU', feature_architecture=architecture)
    model = m.pipeline(recursive_data, save_model=False, plot=False)

    # the expression is part of the model, not of the data
    step_model = model.get_layer('out_model')
    layers = [layer.name for layer in step_model.layers]
    assert any('multiply' in name for name in layers), layers
    assert any('subtract' in name for name in layers), layers
    assert any('batch_normalization' in name for name in layers), layers
    assert 'heat_flow' not in recursive_data.columns

    # and the model config keeps track of it
    config = m.get_config()
    assert config['feature_architecture'][-1].startswith('ModularNormalization')


@pytest.fixture
def warmup_data(file_path):
    """Multi step data whose warmup holds the inputs of the model, which is the default."""
    Logger.setup_logger(base_path=base_path, folder_name='unittests\test_closed_loop', override=True)
    inputs = ['weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u', 'reaTZon_y']
    prep = PreprocessingMultiStep(inputs=inputs, output='reaTZon_y', label_width=4, warmup_width=3)
    assert prep.init_features == inputs, "the warmup should hold the inputs by default"
    return prep.pipeline(file_path)


def test_warmup_with_the_model_itself(warmup_data):
    """By default no separate initialization model is built.

    Instead the model warms its own states up on the warmup sequence, which is what an
    MPC does, so training and MPC see the same thing.
    """
    m = RNNModel(epochs=1, rnn_layer='GRU')
    assert m.init_layer is None, "no initialization model should be built by default"
    model = m.pipeline(warmup_data, save_model=False, plot=False)

    nested = [layer.name for layer in model.layers if isinstance(layer, keras.Model)]
    assert 'out_model' in nested, nested
    assert 'init_model' not in nested, nested

    # the warmup window holds the inputs, directly in front of the prediction window
    main, warm = warmup_data.X_train[0], warmup_data.X_train[1]
    assert warm.shape[-1] == main.shape[-1]

    # and the states it produces are the ones the model was trained with
    step_model = model.get_layer('out_model')
    states = [np.zeros((1, 32), dtype='float32')
              for _ in range(len(step_model.inputs) - 1)]
    for step in range(warm.shape[1]):
        state_input = states[0] if len(states) == 1 else list(states)
        result = step_model([warm[:1, step:step + 1, :], state_input], training=False)
        states = [keras.ops.convert_to_numpy(state) for state in result[1:]]
    state_input = states[0] if len(states) == 1 else list(states)
    from_warmup = keras.ops.convert_to_numpy(
        step_model([main[:1, :1, :], state_input], training=False)[0]
    )
    from_model = model.predict([main[:1], warm[:1]], verbose=0)
    assert np.allclose(from_warmup[:, 0, :], from_model[:, 0, :], atol=1e-4)


def test_warmup_features_have_to_match(file_path):
    """The model can only be applied to the warmup if it holds the same features."""
    Logger.setup_logger(base_path=base_path, folder_name='unittests\test_closed_loop', override=True)
    prep = PreprocessingMultiStep(inputs=['weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u'],
                                  output='reaTZon_y', label_width=4, warmup_width=3,
                                  init_features=['reaTZon_y'])
    td = prep.pipeline(file_path)

    m = RNNModel(epochs=1, rnn_layer='GRU')
    with pytest.raises(ValueError, match="init_features"):
        m.pipeline(td, save_model=False, plot=False)

    # a separate initialization model can use those features
    m = RNNModel(epochs=1, rnn_layer='GRU', init_layer='GRU')
    model = m.pipeline(td, save_model=False, plot=False)
    assert 'init_model' in [layer.name for layer in model.layers]
