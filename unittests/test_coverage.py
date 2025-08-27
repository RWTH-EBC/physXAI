import json
import os
from unittest.mock import patch
import keras
import pytest
######################################################################################################################
from physXAI.utils.logging import Logger, get_parent_working_directory
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep, PreprocessingMultiStep, \
    PreprocessingData
from physXAI.preprocessing.constructed import Feature, FeatureConstruction, FeatureConstant
from physXAI.feature_selection.recursive_feature_elimination import recursive_feature_elimination_pipeline
from physXAI.models import LinearRegressionModel, AbstractModel
from physXAI.models.ann.ann_design import ClassicalANNModel, CMNNModel, LinANNModel, PINNModel, RNNModel, \
    RBFModel


@pytest.fixture(autouse=True)
def disable_plotly_show():
    """Automatically disable plotly show() for all tests"""
    with patch('plotly.graph_objects.Figure.show'):
        yield

@pytest.fixture(scope='module')
def file_path():
    return r"data/bestest_hydronic_heat_pump/pid_data.csv"

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
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=False)
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)

    # Constructed features
    x = Feature('oveHeaPumY_u')
    y = Feature('weaSta_reaWeaTDryBul_y')
    x.lag(2)
    x.lag(4, False)
    z = x + 1
    z.rename('new_name')
    1 + x
    x - 1
    1 - x
    x * 3
    3 * x
    y / 3
    3 / y
    x**2
    x.exp()
    x.sin()
    x.cos()
    FeatureConstant(1, 'name')

    # Create & process Training data
    prep = PreprocessingSingleStep(inputs_php, output_php)
    prep.pipeline(file_path)

def test_preprocessing_multistep(file_path, inputs_tair, output_tair):
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)

    x1 = Feature('reaTZon_y')
    x1.lag(2)  # reaTZon_y_lag1, reaTZon_y_lag2
    x2 = Feature('weaSta_reaWeaTDryBul_y')
    x2.lag(1)  # weaSta_reaWeaTDryBul_y_lag1
    x3 = Feature('oveHeaPumY_u')
    x3.lag(2)  # oveHeaPumY_u_lag1, oveHeaPumY_u_lag2

    # EvaluateMultiStep: Prepare Preprocessing
    prep = PreprocessingMultiStep(inputs_tair, output_tair, 6, 6, init_features=['reaTZon_y'],
                                  overlapping_sequences=False, batch_size=1)
    prep.pipeline(file_path)

@pytest.fixture(scope='module')
def p_hp_data(file_path, inputs_php, output_php):
    # Setup up logger for saving
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)
    # Create & process Training data
    prep = PreprocessingSingleStep(inputs_php, output_php)
    td = prep.pipeline(file_path)
    return prep, td

@pytest.fixture(scope='module')
def tair_data_delta(file_path, inputs_tair, output_tair):
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)
    x1 = Feature('reaTZon_y')
    x1.lag(2)  # reaTZon_y_lag1, reaTZon_y_lag2
    x2 = Feature('weaSta_reaWeaTDryBul_y')
    x2.lag(1)  # weaSta_reaWeaTDryBul_y_lag1
    x3 = Feature('oveHeaPumY_u')
    x3.lag(2)  # oveHeaPumY_u_lag1, oveHeaPumY_u_lag2
    prep = PreprocessingMultiStep(inputs_tair, output_tair, 3, 0, init_features=['reaTZon_y'],
                                  overlapping_sequences=False, batch_size=1)
    td = prep.pipeline(file_path)
    return prep, td

@pytest.fixture(scope='module')
def tair_data_noval(file_path, inputs_tair, output_tair):
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)
    x1 = Feature('reaTZon_y')
    x1.lag(2)  # reaTZon_y_lag1, reaTZon_y_lag2
    x2 = Feature('weaSta_reaWeaTDryBul_y')
    x2.lag(1)  # weaSta_reaWeaTDryBul_y_lag1
    x3 = Feature('oveHeaPumY_u')
    x3.lag(2)  # oveHeaPumY_u_lag1, oveHeaPumY_u_lag2
    prep = PreprocessingMultiStep(inputs_tair, output_tair, 3, 0, init_features=['reaTZon_y'],
                                  overlapping_sequences=False, batch_size=1, val_size=0)
    td = prep.pipeline(file_path)
    return prep, td


@pytest.fixture(scope='module')
def tair_data_total(file_path, inputs_tair, output_tair):
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)
    x1 = Feature('reaTZon_y')
    x1.lag(2)  # reaTZon_y_lag1, reaTZon_y_lag2
    x2 = Feature('weaSta_reaWeaTDryBul_y')
    x2.lag(1)  # weaSta_reaWeaTDryBul_y_lag1
    x3 = Feature('oveHeaPumY_u')
    x3.lag(2)  # oveHeaPumY_u_lag1, oveHeaPumY_u_lag2
    prep = PreprocessingMultiStep(inputs_tair, 'reaTZon_y', 3, 0, init_features=['reaTZon_y'],
                                  overlapping_sequences=False, batch_size=1)
    td = prep.pipeline(file_path)
    return prep, td

def test_model_linReg(inputs_php, output_php, file_path):
    # Setup up logger for saving
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)
    # Create & process Training data
    prep = PreprocessingSingleStep(inputs_php, output_php, val_size=0)
    td = prep.pipeline(file_path)

    # Check Models
    m = LinearRegressionModel()  # Linear Regression
    m.pipeline(td)

    # Log setup
    Logger.log_setup(prep, m, save_name_model='model_linReg.json')
    Logger.save_training_data(td, path=os.path.join(Logger._logger, 'training_data2'))

def test_model_ann(p_hp_data, inputs_php, output_php, file_path):
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
    prep = p_hp_data[0]
    td = p_hp_data[1]

    m = LinANNModel(epochs=1,  n_neurons=4)  # Residual model
    m.pipeline(td, save_model=False, plot=False)

    m = LinANNModel(epochs=1, n_layers=2, n_neurons=[4, 4], activation_function=['softplus', 'softplus'])  # Residual model
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

    # Check PINNs
    # PINN: Prepare preprocessing
    output_php = [output_php] + ['pinn']
    u_hp = Feature('oveHeaPumY_u')
    u_hp_logistic = Feature('Func(logistic)')
    t_amb = Feature('weaSta_reaWeaTDryBul_y')
    TAirRoom = Feature('reaTZon_y')
    pinn = (u_hp * 10000 * ((TAirRoom + 15 - t_amb) / ((TAirRoom + 15) * 0.55)) + (1110 + 500) * u_hp_logistic)
    pinn.rename('pinn')

    # PINN: Preprocessing
    prep = PreprocessingSingleStep(inputs_php, output_php)
    td = prep.pipeline(file_path)
    m = PINNModel(pinn_weights=[1], epochs=1, n_neurons=4)
    m.pipeline(td, save_model=False, plot=False)

    prep = PreprocessingSingleStep(inputs_php, output_php, val_size=0)
    td = prep.pipeline(file_path)
    m = PINNModel(pinn_weights=None, epochs=1, n_neurons=4)
    m.pipeline(td, save_model=True, plot=False)

    m.epochs = 1
    m.online_pipeline(td, os.path.join(Logger._logger, 'model.keras'))

    # Log setup
    Logger.log_setup(prep, m)
    Logger.save_training_data(td)

def test_models_rnn(file_path):
    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)

    # RNN
    inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u']
    inits = ['reaTZon_y']
    output = 'reaTZon_y'
    prep = PreprocessingMultiStep(inputs, output, 4, 2, init_features=inits)
    td = prep.pipeline(file_path)

    m = RNNModel(epochs=1, rnn_layer='LSTM', init_layer='dense')
    m.pipeline(td, os.path.join(Logger._logger, 'model2.keras'))
    Logger.log_setup(td, m, 'preprocessing_config2.json',
                     save_name_constructed='constructed_config2.json')
    Logger.save_training_data(td)

    m.epochs = 1
    m.online_pipeline(td, os.path.join(Logger._logger, 'model2.keras'), plot=False, save_model=False)

    m = RNNModel(epochs=1, rnn_layer='RNN', init_layer='dense')
    m.pipeline(td, save_model=False, plot=False)
    m = RNNModel(epochs=1, rnn_layer='LSTM')
    m.pipeline(td, save_model=False, plot=False)
    m = RNNModel(epochs=1, rnn_layer='GRU')
    m.pipeline(td, save_model=False, plot=False)
    m = RNNModel(epochs=1, rnn_layer='RNN')
    m.pipeline(td, save_model=True, plot=False)

    prep = PreprocessingMultiStep(inputs, output, 4, 0, val_size=0)
    td = prep.pipeline(file_path)
    m = RNNModel(epochs=1, rnn_layer='LSTM', early_stopping_epochs=None)
    m.pipeline(td, save_model=False, plot=False)
    m = RNNModel(epochs=1, rnn_layer='RNN', early_stopping_epochs=None)
    m.pipeline(td, save_model=False, plot=False)

def test_read_setup():

    Logger.setup_logger(folder_name='unittests\\test_coverage', override=True)

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

    save_name_constructed = Logger.save_name_constructed
    path = os.path.join(Logger._logger, save_name_constructed)
    with open(path, "r") as f:
        config_constructed = json.load(f)
    FeatureConstruction.from_config(config_constructed)

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
    monkeypatch.setattr('builtins.input', lambda _: "2")

    prep = p_hp_data[0]

    m = LinearRegressionModel()

    recursive_feature_elimination_pipeline(file_path, prep, m, ascending_lag_order=False)

    monkeypatch.setattr('builtins.input', lambda _: "")
    recursive_feature_elimination_pipeline(file_path, prep, m, ascending_lag_order=True,
                                           fixed_inputs=['weaSta_reaWeaTDryBul_y', 'oveHeaPumY_u'])

def test_feature_selection_multi(monkeypatch, tair_data_delta, tair_data_noval ,tair_data_total, file_path):
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