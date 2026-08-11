import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.training_data import TrainingData, TrainingDataMultiStep

seconds_per_day = 24 * 60 * 60

def _stratify_values_or_none(values: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(values, return_counts=True)

    if len(counts) > 1 and np.all(counts >= 2):
        return values

    return None


def _process_chunks(prep: PreprocessingSingleStep, raw_data: pd.DataFrame, selected_chunk_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.DataFrame] = []
    chunk_id_parts: list[np.ndarray] = []

    for chunk_id in np.sort(selected_chunk_ids):
        raw_chunk = raw_data.loc[raw_data["_chunk_id"] == chunk_id]

        raw_chunk = raw_chunk.drop(columns="_chunk_id").copy()

        x_chunk, y_chunk = prep.process_data(raw_chunk)

        if len(x_chunk) < 2:
            raise ValueError(f"Chunk {chunk_id} contains fewer than two processed samples!")

        time_values = x_chunk.index.to_numpy(dtype=float)

        if not np.allclose(np.diff(time_values), float(prep.time_step)):
            raise ValueError(f"Chunk {chunk_id} contains a time gap!")

        x_parts.append(x_chunk)
        y_parts.append(y_chunk)

        chunk_id_parts.append(np.full(len(x_chunk), chunk_id, dtype=np.int32))

    return(
        pd.concat(x_parts, axis=0),
        pd.concat(y_parts, axis=0),
        np.concatenate(chunk_id_parts),
    )

def _make_multistep_sequences(
        prep: PreprocessingSingleStep,
        raw_data: pd.DataFrame,
        selected_chunk_ids: np.ndarray,
        horizon: int,
        sequence_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_data, y_data, sampled_chunk_ids = _process_chunks(prep=prep, raw_data=raw_data, selected_chunk_ids=selected_chunk_ids)

    x_values = x_data.to_numpy(dtype=np.float32)
    y_values = y_data.to_numpy(dtype=np.float32)

    x_sequences = []
    y_sequences = []
    sequence_chunk_ids = []

    for chunk_id in np.sort(selected_chunk_ids):
        chunk_mask = sampled_chunk_ids == chunk_id

        x_chunk = x_values[chunk_mask]
        y_chunk = y_values[chunk_mask]

        if len(x_chunk) < horizon:
            raise ValueError(f"Chunk {chunk_id + 1} contains only {len(x_chunk)} processed samples, but the requested horizon is {horizon}!")

        for start in range(0, len(x_chunk) - horizon + 1, sequence_stride):
            end = start + horizon

            x_sequences.append(x_chunk[start:end])
            y_sequences.append(y_chunk[start:end])
            sequence_chunk_ids.append(chunk_id)

    if not x_sequences:
        raise ValueError("No Multi-Step sequences could be crated!")

    return (
        np.stack(x_sequences).astype(np.float32),
        np.stack(y_sequences).astype(np.float32),
        np.asarray(sequence_chunk_ids, dtype=np.int32)
    )

def _make_multistep_tf_dataset (x_sequences: np.ndarray, y_sequences: np.ndarray, batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((x_sequences, y_sequences))

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_chunked_training_data(
        prep: PreprocessingSingleStep, 
        file_path: str, 
        chunk_seconds: int = seconds_per_day
) -> TrainingData:
    raw_data = prep.load_data(file_path).copy()

    time_values = raw_data.index.to_numpy(dtype=float)

    if not np.allclose(np.diff(time_values), float(prep.time_step)):
        raise ValueError("raw data contains time gaps!")

    raw_data["_chunk_id"] = ((time_values - time_values[0]) // chunk_seconds).astype(np.int32)

    chunk_sizes = raw_data.groupby("_chunk_id").size()

    expected_chunk_size = chunk_seconds / float(prep.time_step)

    if not expected_chunk_size.is_integer():
        raise ValueError("chunk_seconds must be a multiplier of 'prep.time_step'!")

    if not np.all(chunk_sizes.to_numpy() == int(expected_chunk_size)):
        raise ValueError("At least one chunk is not a complete day!")

    chunk_ids = chunk_sizes.index.to_numpy(dtype=np.int32)
    n_chunks = len(chunk_ids)

    n_test = int(round(n_chunks * prep.test_size))
    n_val = int(round(n_chunks * prep.val_size))
    n_holdout = n_test + n_val

    if (n_test < 1 or n_val < 1 or n_chunks - n_holdout < 1):
        raise ValueError("The requested chunk split is too small!")

    if "weekend_flag" in raw_data.columns:
        labels = raw_data.groupby("_chunk_id")["weekend_flag"].max().loc[chunk_ids].to_numpy(dtype=np.int32)

        stratify_all = _stratify_values_or_none(labels)

    else:
        labels = None
        stratify_all = None

    train_ids, holdout_ids = train_test_split(
        chunk_ids,
        test_size=n_holdout,
        random_state=prep.random_state,
        shuffle=True,
        stratify=stratify_all,
    )

    if labels is not None:
        label_by_chunk = dict(zip(chunk_ids, labels))

        holdout_label = np.array(
            [
                label_by_chunk[chunk_id]
                for chunk_id in holdout_ids
            ],
            dtype=np.int32
        )

        stratify_holdout = _stratify_values_or_none(holdout_label)

    else:
        stratify_holdout = None

    val_ids, test_ids = train_test_split(
        holdout_ids, 
        test_size=n_test,
        random_state=prep.random_state,
        shuffle=True,
        stratify=stratify_holdout,
    )

    x_train, y_train, train_chunk_ids = _process_chunks(
        prep=prep,
        raw_data=raw_data,
        selected_chunk_ids=train_ids,
    )

    x_val, y_val, val_chunk_ids = _process_chunks(
       prep=prep,
       raw_data=raw_data,
       selected_chunk_ids=val_ids,
    ) 

    x_test, y_test, test_chunk_ids = _process_chunks(
        prep=prep,
        raw_data=raw_data,
        selected_chunk_ids=test_ids,
    )

    td = TrainingData(
        X_train=x_train.to_numpy(),
        X_val=x_val.to_numpy(),
        X_test=x_test.to_numpy(),
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        columns=x_train.columns.to_list(),
    )

    td.train_chunk_ids = train_chunk_ids
    td.val_chunk_ids =  val_chunk_ids
    td.test_chunk_ids = test_chunk_ids

    td.split_chunk_numbers = {
        "train": (np.sort(train_ids) + 1).tolist(),
        "validation": (np.sort(val_ids) + 1).tolist(),
        "test": (np.sort(test_ids) + 1).tolist(),
    }

    td.time_train = x_train.index.to_numpy(dtype=float)
    td.time_val = x_val.index.to_numpy(dtype=float)
    td.time_test = x_test.index.to_numpy(dtype=float)

    td.add_file_path(file_path)

    return td

def make_chunked_multistep_data(
        prep: PreprocessingSingleStep,
        file_path: str,
        split_chunk_numbers: dict[str, list[int]],
        horizon: int,
        recursive_output_column: str,
        chunk_seconds: int = seconds_per_day,
        sequence_stride: int | None = None,
        batch_size: int = 32,
) -> TrainingDataMultiStep:
    """
    
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    if sequence_stride is None:
        sequence_stride = horizon

    if sequence_stride < 1:
        raise ValueError("sequence_stride must be at least 1 or None!")

    if batch_size < 1:
        raise ValueError("bacth_size must be at least 1!")

    if recursive_output_column not in prep.inputs:
        raise ValueError("The recursive output column is not contained in prep.inputs!")

    required_split_names = {"train", "validation", "test"}

    missing_split_names = required_split_names - set(split_chunk_numbers)

    if missing_split_names:
        raise ValueError(f"Missing chunk splits: {sorted(missing_split_names)}")

    raw_data = prep.load_data(file_path).copy()

    time_values = raw_data.index.to_numpy(dtype=float)

    if not np.allclose(np.diff(time_values), float(prep.time_step)):
        raise ValueError("Raw data contains time gaps!")

    raw_data["_chunk_id"] = ((time_values - time_values[0]) // chunk_seconds).astype(np.int32)

    chunk_sizes = raw_data.groupby("_chunk_id").size()

    expected_chunk_size = chunk_seconds / float(prep.time_step)

    if not expected_chunk_size.is_integer():
        raise ValueError("chunk_seconds must be a multiple of prep.time_step!")

    if not np.all(chunk_sizes.to_numpy() == int(expected_chunk_size)):
        raise ValueError("At least one chunk is incomplete!")

    available_chunk_ids = chunk_sizes.index.to_numpy(dtype=np.int32)

    available_chunk_numbers = set((available_chunk_ids + 1).tolist())

    split_numbers = {}

    for split_name in ("train", "validation", "test"):
        numbers = np.asarray(split_chunk_numbers[split_name], dtype=np.int32)

        if len(numbers) == 0:
            raise ValueError(f"The split {split_name} does not contain any chunks!")

        split_numbers[split_name] = numbers

    all_selected_numbers = np.concatenate(
        [
            split_numbers["train"],
            split_numbers["validation"],
            split_numbers["test"],
        ]
    )

    if len(np.unique(all_selected_numbers)) != len(all_selected_numbers):
        raise ValueError("A chunk occurs in more then one split!")

    selected_number_set = set(all_selected_numbers.tolist())

    unkown_numbers = selected_number_set - available_chunk_numbers

    missing_numbers = available_chunk_numbers - selected_number_set

    if unkown_numbers: 
        raise ValueError(f"Unknown chunk numbers were passed: {sorted(unkown_numbers)}!")

    if missing_numbers:
        raise ValueError(f"The following chunks were not assigned to a split: {sorted(missing_numbers)}!")

    train_ids = split_numbers["train"] - 1
    val_ids = split_numbers["validation"] - 1
    test_ids = split_numbers["test"] - 1

    x_train, y_train, train_sequences_chunk_ids = _make_multistep_sequences(
        prep=prep,
        raw_data=raw_data,
        selected_chunk_ids=train_ids,
        horizon=horizon,
        sequence_stride=sequence_stride,
    )

    x_val, y_val, val_sequences_chunk_ids = _make_multistep_sequences(
        prep=prep,
        raw_data=raw_data,
        selected_chunk_ids=val_ids,
        horizon=horizon,
        sequence_stride=sequence_stride,
    )

    x_test, y_test, test_sequences_chunk_ids = _make_multistep_sequences(
        prep=prep,
        raw_data=raw_data,
        selected_chunk_ids=test_ids,
        horizon=horizon,
        sequence_stride=sequence_stride,
    )

    train_ds = _make_multistep_tf_dataset(
        x_sequences=x_train,
        y_sequences=y_train,
        batch_size=batch_size,
    )

    val_ds = _make_multistep_tf_dataset(
        x_sequences=x_val,
        y_sequences=y_val,
        batch_size=batch_size,
    )

    test_ds = _make_multistep_tf_dataset(
        x_sequences=x_test,
        y_sequences=y_test,
        batch_size=batch_size,
    )

    td = TrainingDataMultiStep(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        columns=prep.inputs,
        output=prep.output,
        init_columns=[recursive_output_column],
    )

    td.train_sequence_chunk_ids = train_sequences_chunk_ids
    td.val_sequences_chunk_ids = val_sequences_chunk_ids
    td.test_sequences_chunk_ids = test_sequences_chunk_ids

    td.split_chunk_numbers = {
        "train": split_numbers["train"].tolist(),
        "validation": split_numbers["validation"].tolist(),
        "test": split_numbers["test"].tolist(),
    }

    td.horizon = horizon
    td.sequence_stride = sequence_stride

    td.add_file_path(file_path)

    return td