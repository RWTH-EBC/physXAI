import json
import os
import pathlib
from unittest.mock import patch
import keras
import pytest
######################################################################################################################
from physXAI.utils.logging import Logger, get_parent_working_directory
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep, PreprocessingMultiStep, \
    PreprocessingData
from physXAI import expr as px
from physXAI.feature_selection.recursive_feature_elimination import recursive_feature_elimination_pipeline
from physXAI.models.models import LinearRegressionModel, AbstractModel
from physXAI.models.ann.ann_design import ClassicalANNModel, CMNNModel, LinANNModel, PINNModel, RNNModel, \
    RBFModel


base_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'stored_data')


@pytest.fixture(autouse=True)
def disable_plotly_show():
    """Automatically disable plotly show() for all tests"""
    with patch('plotly.graph_objects.Figure.show'):
        yield

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

def test_path_setup():
    get_parent_working_directory()

def test_preprocessing(monkeypatch, file_path, inputs_php, output_php):
    monkeypatch.setattr('builtins.input', lambda _: "Y")

    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=False)
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

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
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)


    # Sequences of a recurrent model
    prep = PreprocessingMultiStep(inputs=inputs_tair, output=px.Change('reaTZon_y'), label_width=6, warmup_width=6,
                                  overlapping_sequences=False, eval_horizon='1h', time_index_col='time')
    td = prep.pipeline(file_path)
    assert td.X_train.shape[1] == 12 and td.eval_horizon == 4

@pytest.fixture(scope='module')
def p_hp_data(file_path, inputs_php, output_php):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)
    # Create & process Training data
    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php)
    td = prep.pipeline(file_path)
    return prep, td

@pytest.fixture(scope='module')
def tair_data_delta(file_path, inputs_tair, output_tair):
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)
    # single step model, evaluated over a horizon of 3 steps
    prep = PreprocessingSingleStep(inputs=inputs_tair, output='Change(reaTZon_y)', eval_horizon=3)
    td = prep.pipeline(file_path)
    return prep, td

@pytest.fixture(scope='module')
def tair_data_noval(file_path, inputs_tair, output_tair):
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)
    prep = PreprocessingSingleStep(inputs=inputs_tair, output='Change(reaTZon_y)', eval_horizon=3, val_size=0)
    td = prep.pipeline(file_path)
    return prep, td


@pytest.fixture(scope='module')
def tair_data_total(file_path, inputs_tair, output_tair):
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)
    prep = PreprocessingSingleStep(inputs=inputs_tair, output='reaTZon_y', eval_horizon=3)
    td = prep.pipeline(file_path)
    return prep, td

def test_model_linReg(inputs_php, output_php, file_path):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

    # Create & process Training data
    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php, val_size=0)
    td = prep.pipeline(file_path)

    # Check Models
    m = LinearRegressionModel()  # Linear Regression
    m.pipeline(td)

    # Log setup
    Logger.log_setup(prep, m, save_name_model='model_linReg.json')
    Logger.save_training_data(td, path=os.path.join(Logger._logger, 'training_data2'))

def test_model_ann(p_hp_data, inputs_php, output_php, file_path):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

    prep = p_hp_data[0]
    td = p_hp_data[1]

    m = ClassicalANNModel(epochs=1, n_neurons=[4, 4], n_layers=2, activation_function=['softplus', 'softplus'],
                          early_stopping_epochs=None, rescale_output=False)
    m.pipeline(td)

    m.epochs = 1
    m.online_pipeline(td, os.path.join(Logger._logger, 'model.keras'))

    # Log setup
    Logger.log_setup(None, m)
    Logger.save_training_data(td)

def test_model_cmnn(p_hp_data, inputs_php, output_php, file_path):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

    prep = p_hp_data[0]
    td = p_hp_data[1]

    m = CMNNModel(n_neurons=4, monotonies={  # CMNN
        'oveHeaPumY_u': 1,
        'Func(logistic)': 1,
        'weaSta_reaWeaTDryBul_y': -1,
        'reaTZon_y': 1,
    }, activation_split=[1, 1, 1],  # Proportions for splitting neurons into convex, concave, and saturated activation
    epochs=1)
    m.pipeline(td, save_model=True, plot=False, save_path=os.path.join(Logger._logger, 'model2.keras'))
    keras.saving.load_model(os.path.join(Logger._logger, 'model2.keras'))
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

    # Log setup
    Logger.log_setup(prep, m)
    Logger.save_training_data(td)

def test_model_linANN(p_hp_data, inputs_php, output_php, file_path):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

    prep = p_hp_data[0]
    td = p_hp_data[1]

    m = LinANNModel(epochs=1,  n_neurons=4)  # Residual model
    m.pipeline(td, save_model=False, plot=False)

    # Log setup
    Logger.log_setup(prep, m)
    Logger.save_training_data(td)

    m = RBFModel(epochs=1, n_neurons=4, rescale_output=False)
    m.pipeline(td, save_model=True, plot=False, save_path=os.path.join(Logger._logger, 'model2.keras'))
    keras.saving.load_model(os.path.join(Logger._logger, 'model2.keras'))

    # Log setup
    Logger.log_setup(prep, m)
    Logger.save_training_data(td)

def test_model_pinn(inputs_php, output_php, file_path):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

    # Check PINNs
    # PINN: Prepare preprocessing
    output_php = [output_php] + ['pinn']
    v = px.Vars()
    u_hp, u_hp_logistic = v.oveHeaPumY_u, v['Func(logistic)']
    t_amb, TAirRoom = v.weaSta_reaWeaTDryBul_y, v.reaTZon_y
    pinn = (u_hp * 10000 * ((TAirRoom + 15 - t_amb) / ((TAirRoom + 15) * 0.55)) + (1110 + 500) * u_hp_logistic)

    # PINN: Preprocessing
    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php, columns={'pinn': pinn})
    td = prep.pipeline(file_path)
    m = PINNModel(pinn_weights=[1], epochs=1, n_neurons=4)
    m.pipeline(td, save_model=False, plot=False)

    prep = PreprocessingSingleStep(inputs=inputs_php, output=output_php, val_size=0, columns={'pinn': pinn})
    td = prep.pipeline(file_path)
    m = PINNModel(pinn_weights=None, epochs=1, n_neurons=4)
    m.pipeline(td, save_model=True, plot=False)

    m.epochs = 1
    m.online_pipeline(td, os.path.join(Logger._logger, 'model.keras'))

    # Log setup
    Logger.log_setup(prep, m)
    Logger.save_training_data(td)

def test_models_rnn(file_path):
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

    # RNN predicting the change of the air temperature, warmed up on the measured past
    inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u']
    prep = PreprocessingMultiStep(inputs=inputs, output=px.Change('reaTZon_y'), label_width=4, warmup_width=2)
    td = prep.pipeline(file_path)

    m = RNNModel(epochs=1, rnn_layer='LSTM', rnn_units=8)
    m.pipeline(td, os.path.join(Logger._logger, 'model2.keras'))
    Logger.log_setup(prep, m, 'preprocessing_config2.json')
    Logger.save_training_data(td)

    m.epochs = 1
    m.online_pipeline(td, os.path.join(Logger._logger, 'model2.keras'), plot=False, save_model=False)

    for layer in ('RNN', 'GRU'):
        m = RNNModel(epochs=1, rnn_layer=layer, rnn_units=8)
        m.pipeline(td, save_model=False, plot=False)

    # closed loop: the air temperature is an input, too, and without validation data
    prep = PreprocessingMultiStep(inputs=[*inputs, 'reaTZon_y'], output=px.Change('reaTZon_y'), label_width=4,
                                  warmup_width=0, val_size=0)
    td = prep.pipeline(file_path)
    m = RNNModel(epochs=1, rnn_layer='GRU', rnn_units=8, early_stopping_epochs=None)
    m.pipeline(td, save_model=True, plot=False)

def test_read_setup():

    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)

    # Read setup
    save_name_preprocessing = Logger.save_name_preprocessing
    path = os.path.join(Logger._logger, save_name_preprocessing)
    with open(path, "r") as f:
        config_prep = json.load(f)
    PreprocessingData.from_config(config_prep)

    save_name_preprocessing = 'preprocessing_config2.json'
    path = os.path.join(Logger._logger, save_name_preprocessing)
    with open(path, "r") as f:
        config_prep = json.load(f)
    PreprocessingData.from_config(config_prep)

    save_name_model = Logger.save_name_model_config
    path = os.path.join(Logger._logger, save_name_model)
    with open(path, "r") as f:
        config_model = json.load(f)
    AbstractModel.model_from_config(config_model)

    save_name_model = 'model_linReg.json'
    path = os.path.join(Logger._logger, save_name_model)
    with open(path, "r") as f:
        config_model = json.load(f)
    AbstractModel.model_from_config(config_model)

def test_feature_selection(monkeypatch, p_hp_data, file_path):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)
    monkeypatch.setattr('builtins.input', lambda _: "2")

    prep = p_hp_data[0]

    m = LinearRegressionModel()

    recursive_feature_elimination_pipeline(file_path, prep, m, ascending_lag_order=False)

    monkeypatch.setattr('builtins.input', lambda _: "")
    recursive_feature_elimination_pipeline(file_path, prep, m, ascending_lag_order=True,
                                           fixed_inputs=['weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u'])

def test_feature_selection_multi(monkeypatch, tair_data_delta, tair_data_noval ,tair_data_total, file_path):
    # Setup up logger for saving
    Logger.setup_logger(base_path=base_path, folder_name='unittests/test_coverage', override=True)
    monkeypatch.setattr('builtins.input', lambda _: "2")

    prep = tair_data_delta[0]
    td = tair_data_delta[1]

    prep2 = tair_data_total[0]
    td2 = tair_data_total[1]

    prep3 = tair_data_noval[0]
    td3 = tair_data_noval[1]

    # EvaluateMultiStep
    m = LinearRegressionModel()

    # Feature Selection
    recursive_feature_elimination_pipeline(file_path, prep, m, use_multi_step_error=True)
    recursive_feature_elimination_pipeline(file_path, prep2, m, use_multi_step_error=True)
    m.pipeline(td2, save_model=False, plot=False)
    m.pipeline(td3, save_model=True, save_path=os.path.join(Logger._logger, 'model2.joblib'))
    m.load_model(os.path.join(Logger._logger, 'model.joblib'))

    m = ClassicalANNModel(epochs=1, n_neurons=4)
    recursive_feature_elimination_pipeline(file_path, prep, m, use_multi_step_error=False)
    m.pipeline(td3, save_model=False)
    m = ClassicalANNModel(epochs=1, n_neurons=4)
    recursive_feature_elimination_pipeline(file_path, prep2, m, use_multi_step_error=False)
    m.pipeline(td2, save_model=False, plot=False)
    Logger.log_setup(prep, None)
    Logger.save_training_data(td, path=os.path.join(Logger._logger, 'training_data2.json'))