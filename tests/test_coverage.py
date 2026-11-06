import json
import os
import pathlib
from unittest.mock import patch

import keras
import pytest

from physXAI import expr as px
from physXAI.feature_selection.selection import FeatureSelection

######################################################################################################################
from physXAI.io import Run
from physXAI.models.ann.ann_design import ClassicalANNModel, CMNNModel, LinANNModel, PINNModel, RBFModel, RNNModel
from physXAI.models.models import AbstractModel, LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingData, PreprocessingMultiStep, PreprocessingSingleStep


@pytest.fixture(autouse=True)
def disable_plotly_show():
    """Automatically disable plotly show() for all tests"""
    with patch('plotly.graph_objects.Figure.show'):
        yield


@pytest.fixture
def run(tmp_path):
    """Every test writes into its own run directory."""
    return Run(tmp_path, 'test_coverage', overwrite=True)


@pytest.fixture(scope='module')
def file_path():
    return os.path.join(pathlib.Path(__file__).resolve().parent.parent, "data/bestest_hydronic_heat_pump/pid_data.csv")

@pytest.fixture(scope='module')
def inputs_php():
    return ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']

@pytest.fixture(scope='module')
def inputs_tair():
    return ['reaTZon_y', 'weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u', 'oveHeaPumY_u_lag1']

@pytest.fixture(scope='module')
def output_php():
    return 'reaPHeaPum_y'

@pytest.fixture(scope='module')
def output_tair():
    return 'Change(T_zone)'


def check_saved_configs(run: Run, name: str):
    """The saved configs recreate the preprocessing and the model."""
    with open(run.path / f'{name}_preprocessing.json') as f:
        PreprocessingData.from_config(json.load(f))
    with open(run.path / f'{name}_model.json') as f:
        AbstractModel.model_from_config(json.load(f))


def test_preprocessing(file_path, inputs_php, output_php):
    # Derived data columns (expressions evaluated with pandas)
    v = px.Vars()
    x, y = v.oveHeaPumY_u, v.weaSta_reaWeaTDryBul_y
    columns = {'new_name': x + 1, 'a': 1 + x, 'b': x - 1, 'c': 1 - x, 'd': x * 3, 'e': 3 * x, 'f': y / 3, 'g': 3 / y,
               'h': x ** 2, 'i': px.exp(x), 'j': x.lag(4), 'k': px.Const(1.0)}
    prep = PreprocessingSingleStep(inputs=inputs_php + ['new_name', 'j'], output=output_php, columns=columns)
    prep.pipeline(file_path)

    # Create & process Training data
    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php)
    prep.pipeline(file_path)

def test_preprocessing_multistep(file_path, inputs_tair, output_tair):
    # Sequences of a recurrent model
    prep = PreprocessingMultiStep(inputs=inputs_tair, output=px.Change('reaTZon_y'), label_width=6, warmup_width=6,
                                  overlapping_sequences=False, eval_horizon='1h', time_index_col='time')
    td = prep.pipeline(file_path)
    assert td.X_train.shape[1] == 12 and td.eval_horizon == 4

@pytest.fixture(scope='module')
def p_hp_data(file_path, inputs_php, output_php):
    # Create & process Training data
    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php)
    td = prep.pipeline(file_path)
    return prep, td

@pytest.fixture(scope='module')
def tair_data_delta(file_path, inputs_tair, output_tair):
    # single step model, evaluated over a horizon of 3 steps
    prep = PreprocessingSingleStep(inputs=inputs_tair, output='Change(reaTZon_y)', eval_horizon=3)
    td = prep.pipeline(file_path)
    return prep, td

@pytest.fixture(scope='module')
def tair_data_noval(file_path, inputs_tair, output_tair):
    prep = PreprocessingSingleStep(inputs=inputs_tair, output='Change(reaTZon_y)', eval_horizon=3, val_size=0)
    td = prep.pipeline(file_path)
    return prep, td


@pytest.fixture(scope='module')
def tair_data_total(file_path, inputs_tair, output_tair):
    prep = PreprocessingSingleStep(inputs=inputs_tair, output='reaTZon_y', eval_horizon=3)
    td = prep.pipeline(file_path)
    return prep, td

def test_model_linReg(inputs_php, output_php, file_path, run):
    # Create & process Training data
    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php, val_size=0)
    td = prep.pipeline(file_path)

    # Check Models
    m = LinearRegressionModel()  # Linear Regression
    m.pipeline(td)

    files = run.save('linReg', prep, m, td)
    assert set(files) == {'preprocessing', 'model_config', 'training_data', 'model', 'plot'}
    check_saved_configs(run, 'linReg')
    m.load_model(files['model'])

def test_model_ann(p_hp_data, inputs_php, output_php, file_path, run):
    prep = p_hp_data[0]
    td = p_hp_data[1]

    m = ClassicalANNModel(epochs=1, n_neurons=[4, 4], n_layers=2, activation_function=['softplus', 'softplus'],
                          early_stopping_epochs=None, rescale_output=False)
    m.pipeline(td, save_path=str(run.path / 'model.keras'))

    m.epochs = 1
    m.online_pipeline(td, str(run.path / 'model.keras'))

    run.save('ann', prep, m, td)
    check_saved_configs(run, 'ann')

def test_model_cmnn(p_hp_data, inputs_php, output_php, file_path, run):
    prep = p_hp_data[0]
    td = p_hp_data[1]

    m = CMNNModel(n_neurons=4, monotonies={  # CMNN
        'oveHeaPumY_u': 1,
        'Func(logistic)': 1,
        'weaSta_reaWeaTDryBul_y': -1,
        'reaTZon_y': 1,
    }, activation_split=[1, 1, 1],  # Proportions for splitting neurons into convex, concave, and saturated activation
    epochs=1)
    m.pipeline(td, save_model=True, plot=False, save_path=str(run.path / 'model2.keras'))
    keras.saving.load_model(run.path / 'model2.keras')
    m = CMNNModel(n_layers=2, n_neurons=[4, 4], activation_function=['softplus', 'softplus'], monotonies={  # CMNN
        'oveHeaPumY_u': 1,
        'Func(logistic)': 1,
        'weaSta_reaWeaTDryBul_y': -1,
        'reaTZon_y': 1,
    }, activation_split=None,  # Proportions for splitting neurons into convex, concave, and saturated activation
    epochs=1)
    m.pipeline(td, save_model=False, plot=False)
    m = CMNNModel(n_layers=2, n_neurons=[4, 4], activation_function=['softplus', 'softplus'], rescale_output=False,
                  monotonies={  # CMNN
        'oveHeaPumY_u': 1,
        'Func(logistic)': 1,
        'weaSta_reaWeaTDryBul_y': -1,
        'reaTZon_y': 1,
    }, activation_split=[0, 0, 1],  # Proportions for splitting neurons into convex, concave, and saturated activation
    epochs=1)
    m.pipeline(td, save_model=False, plot=False)

    run.save('cmnn', prep, m, td, plot=False)
    check_saved_configs(run, 'cmnn')

def test_model_linANN(p_hp_data, inputs_php, output_php, file_path, run):
    prep = p_hp_data[0]
    td = p_hp_data[1]

    m = LinANNModel(epochs=1,  n_neurons=4)  # Residual model
    m.pipeline(td, save_model=False, plot=False)
    run.save('linANN', prep, m, td, plot=False)
    check_saved_configs(run, 'linANN')

    m = RBFModel(epochs=1, n_neurons=4, rescale_output=False)
    m.pipeline(td, save_model=True, plot=False, save_path=str(run.path / 'model2.keras'))
    keras.saving.load_model(run.path / 'model2.keras')
    run.save('rbf', prep, m, td, plot=False)
    check_saved_configs(run, 'rbf')

def test_model_pinn(inputs_php, output_php, file_path, run):
    # Check PINNs
    # PINN: Prepare preprocessing
    v = px.Vars()
    u_hp, u_hp_logistic = v.oveHeaPumY_u, v['Func(logistic)']
    t_amb, TAirRoom = v.weaSta_reaWeaTDryBul_y, v.reaTZon_y
    pinn = (u_hp * 10000 * ((TAirRoom + 15 - t_amb) / ((TAirRoom + 15) * 0.55)) + (1110 + 500) * u_hp_logistic)

    # PINN: Preprocessing
    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php, columns={'pinn': pinn})
    m = PINNModel(physics_targets={'pinn': 1.0}, epochs=1, n_neurons=4)
    td = prep.pipeline(file_path, model=m)
    m.pipeline(td, save_model=False, plot=False)

    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php, val_size=0, columns={'pinn': pinn})
    m = PINNModel(physics_targets={'pinn': 0.5}, normalize_terms=False, early_stopping_on='total', epochs=1,
                  n_neurons=4)
    td = prep.pipeline(file_path, model=m)
    m.pipeline(td, save_model=True, plot=False, save_path=str(run.path / 'model.keras'))

    m.epochs = 1
    m.online_pipeline(td, str(run.path / 'model.keras'))

    run.save('pinn', prep, m, td)

def test_models_rnn(file_path, run):
    # RNN predicting the change of the air temperature, warmed up on the measured past
    inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u']
    prep = PreprocessingMultiStep(inputs=inputs, output=px.Change('reaTZon_y'), label_width=4, warmup_width=2)
    td = prep.pipeline(file_path)

    m = RNNModel(epochs=1, rnn_layer='LSTM', rnn_units=8)
    m.pipeline(td, str(run.path / 'model2.keras'))
    run.save('rnn', prep, m, td)
    check_saved_configs(run, 'rnn')

    m.epochs = 1
    m.online_pipeline(td, str(run.path / 'model2.keras'), plot=False, save_model=False)

    for layer in ('RNN', 'GRU'):
        m = RNNModel(epochs=1, rnn_layer=layer, rnn_units=8)
        m.pipeline(td, save_model=False, plot=False)

    # closed loop: the air temperature is an input, too, and without validation data
    prep = PreprocessingMultiStep(inputs=[*inputs, 'reaTZon_y'], output=px.Change('reaTZon_y'), label_width=4,
                                  warmup_width=0, val_size=0)
    td = prep.pipeline(file_path)
    m = RNNModel(epochs=1, rnn_layer='GRU', rnn_units=8, early_stopping_epochs=None)
    m.pipeline(td, save_model=True, plot=False, save_path=str(run.path / 'model3'))
    assert (run.path / 'model3.keras').exists()

def test_feature_selection(p_hp_data, file_path, run):
    prep = p_hp_data[0]

    m = LinearRegressionModel()

    FeatureSelection(prep, m, ascending_lag_order=False).run(file_path)

    result = FeatureSelection(prep, m, ascending_lag_order=True,
                              fixed_inputs=['weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u']).run(file_path)
    result.save(run)
    assert (run.path / 'feature_selection.json').exists() and (run.path / 'feature_selection.html').exists()

def test_feature_selection_multi(tair_data_delta, tair_data_noval, tair_data_total, file_path, run):
    prep = tair_data_delta[0]
    td = tair_data_delta[1]

    prep2 = tair_data_total[0]
    td2 = tair_data_total[1]

    td3 = tair_data_noval[1]

    # EvaluateMultiStep
    m = LinearRegressionModel()

    # Feature Selection
    FeatureSelection(prep, m).run(file_path)
    FeatureSelection(prep2, m).run(file_path)
    m.pipeline(td2, save_model=False, plot=False)
    m.pipeline(td3, save_model=True, save_path=str(run.path / 'model2.joblib'))
    m.load_model(run.path / 'model2.joblib')

    m = ClassicalANNModel(epochs=1, n_neurons=4)
    FeatureSelection(prep, m, criterion='single_step').run(file_path)
    m.pipeline(td3, save_model=False)
    m = ClassicalANNModel(epochs=1, n_neurons=4)
    FeatureSelection(prep2, m, criterion='single_step').run(file_path)
    m.pipeline(td2, save_model=False, plot=False)
    run.save('training_data2', prep, None, td)
