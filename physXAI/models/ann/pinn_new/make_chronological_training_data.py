import numpy as np

from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.training_data import TrainingData

def make_chronological_training_data(prep: PreprocessingSingleStep, file_path: str) -> TrainingData:
    df = prep.load_data(file_path)
    X, y = prep.process_data(df)

    time_values = X.index.to_numpy(dtype=float)
    time_differences = np.diff(time_values)

    if not np.allclose(time_differences, float(prep.time_step)):
        raise ValueError("The processed Ghokale data contain time gaps. No (x_k, x_k1) pairs were created!")

    n_total = len(X)
    n_test = int(n_total * prep.test_size)
    n_val = int(n_total * prep.val_size)
    n_train = n_total - n_val - n_test

    if n_train < 2:
        raise ValueError("Not enough data for Gokhale!")

    if n_val == 1:
        raise ValueError("The validation data must contain zero or at least two rows!")

    if n_test < 2:
        raise ValueError("At least two chronological test rows are required!")

    X_train = X.iloc[:n_train].to_numpy()
    y_train = y.iloc[:n_train].to_numpy()

    if n_val > 0:
        X_val = X.iloc[n_train:n_train + n_val].to_numpy()
        y_val = y.iloc[n_train:n_train + n_val].to_numpy()
    else:
        X_val = None
        y_val = None

    X_test = X.iloc[n_train + n_val:].to_numpy()
    y_test = y.iloc[n_train + n_val:].to_numpy()

    td = TrainingData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        columns=X.columns.values.tolist(),
    )

    td.add_file_path(file_path)

    return td