import os
import numpy as np
import pandas as pd
from pathlib import Path
from physXAI.models.modular.modular_expression import ModularTrainable
from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.models.modular.modular_ann import ModularANN, ModularAverage, ModularLinear, ModularModel
from physXAI.utils.logging import Logger
from physXAI.preprocessing.constructed import Feature
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


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
    Logger.setup_logger(base_path=os.path.abspath('models'), folder_name='001', override=True)

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

    out = ModularLinear([
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
        m8
    ])
    m = ModularANN(architecture=out, epochs=1000, random_seed=random_seed)
    model = m.pipeline(td, plot=False, save_model=False)

    os.makedirs('models', exist_ok=True)
    model.save('models/model.keras')


if __name__ == "__main__":
    test_generate_sample_model()
    test_generate_sample_model()