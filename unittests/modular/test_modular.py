import os
import numpy as np
import pandas as pd
import json
import copy
from pathlib import Path
from physXAI.models.modular.modular_expression import ModularTrainable, ModularExpression
from physXAI.models.ann.ann_design import ClassicalANNModel, CMNNModel
from physXAI.models.modular.modular_ann import (ModularANN, ModularAverage, ModularLinear, ModularModel,
                                                ModularExistingModel, ModularMonotoneLinear, ModularPolynomial,
                                                ModularNormalization)
from physXAI.utils.logging import Logger
from physXAI.preprocessing.constructed import Feature
from physXAI.models.models import AbstractModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

base_path = os.path.join(Path(__file__).resolve().parent.parent.parent, 'stored_data')


def test_generate_sample_csv(output_path: str = "data/sample_data.csv", num_rows: int = 1200, num_features: int = 4, seed: int = 42, value_range: tuple = (-100, 100)):
    np.random.seed(seed)
    
    columns = [f"x{i}" for i in range(1, num_features + 1)]
    
    data = {}
    
    for col in columns:
        data[col] = np.random.uniform(value_range[0], value_range[1], num_rows)

    data_with_index = {"": range(num_rows)}
    data_with_index.update(data)
    
    df = pd.DataFrame(data_with_index)
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, sep=";", index=False)

    print(f"Sample CSV file generated at: {output_path}")


def test_generate_sample_model(random_seed: int = 42, training_data_path: str = "data/sample_data.csv"):
    Logger.setup_logger(base_path=base_path, folder_name='unittests\\test_modular', override=True)

    inputs = [f"x{i}" for i in range(1, 4)]
    output = "x4"
    
    features = list()
    for inp in inputs:
        features.append(Feature(inp))

    prep = PreprocessingSingleStep(inputs=inputs, output=output, random_state=random_seed)
    td = prep.pipeline(training_data_path)

    # TODO: Flatten, BatchNorm, Cropping1D, Reshape, RBF

    m1 = ModularModel(ClassicalANNModel(random_seed=random_seed), inputs=features)
    m2 = ModularTrainable(initial_value=0.5)
    mX = ModularTrainable(initial_value=5)
    mY = ModularTrainable(initial_value=0.5)
    m3 = mX + mY
    m4 = mX - mY
    m5 = mX * mY
    m6 = mX / mY
    m7 = mX ** mY
    m8 = ModularAverage([mX, mY])

    # Existing model
    cmnn = CMNNModel(monotonies={'x1': 1, 'x2': -1, 'x3': 0}, activation_split=[1, 1, 1], epochs=50)
    cmnn_model = cmnn.pipeline(td, save_model=False, plot=False)
    me = ModularExistingModel(model=cmnn_model, original_inputs=features, trainable=False)

    mml = ModularMonotoneLinear(inputs=[m3, m4], monotonicities={m3.name: 1, m4.name: -1})
    mp = ModularPolynomial(inputs=[m5, m7, m8], degree=3)
    mn = ModularNormalization(input=m2)

    out = ModularLinear([
        m1,
        m6,
        me,
        mml,
        mp,
        mn,
    ])

    m = ModularANN(architecture=out, epochs=50, random_seed=random_seed)
    model = m.pipeline(td, plot=True, save_model=True)

    Logger.log_setup(preprocessing=prep, model=m)


def test_read_setup(training_data_path: str = "data/sample_data.csv"):
    Logger.setup_logger(base_path=base_path, folder_name='unittests\\test_modular', override=True)

    # Read setup
    save_name_preprocessing = Logger.save_name_preprocessing
    path = os.path.join(Logger._logger, save_name_preprocessing)
    with open(path, "r") as f:
        config_prep = json.load(f)
    prep = PreprocessingSingleStep.from_config(config_prep)

    save_name_modular_expression = Logger.save_name_modular_expression_config
    path = os.path.join(Logger._logger, save_name_modular_expression)
    with open(path, "r") as f:
        modular_expression_config = json.load(f)
    stored_config = copy.deepcopy(modular_expression_config)
    ModularExpression.from_config(modular_expression_config)
    assert check_lists_equal(stored_config, ModularExpression.get_config())

    save_name_model = Logger.save_name_model_config
    path = os.path.join(Logger._logger, save_name_model)
    with open(path, "r") as f:
        config_model = json.load(f)
    stored_config = copy.deepcopy(config_model)
    m = AbstractModel.model_from_config(config_model)
    assert check_lists_equal(stored_config, m.get_config())

    td = prep.pipeline(training_data_path)
    model = m.pipeline(td, plot=True, save_model=True)


def check_lists_equal(list1, list2):
    """Check if all elements in list1 exist and are equal to those in list2."""

    def make_hashable(d):
        """Convert dictionary values to hashable types."""
        if isinstance(d, dict):
            return frozenset((k, make_hashable(v)) for k, v in d.items())
        elif isinstance(d, list):
            return tuple(make_hashable(i) for i in d)
        elif hasattr(d, '__dict__'):  # Check if it's an object with attributes
            return frozenset((key, make_hashable(value)) for key, value in d.__dict__.items())
        else:
            return d  # Return as is if it's already hashable

    set1 = {make_hashable(d) for d in list1}
    set2 = {make_hashable(d) for d in list2}

    return set1 == set2


if __name__ == "__main__":
    test_generate_sample_model()
    test_generate_sample_model()
    test_read_setup()
    