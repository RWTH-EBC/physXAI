import copy
import json
import os
import shutil
from datetime import datetime
import git
from physXAI.preprocessing.constructed import FeatureConstruction
import pickle
from pathlib import Path

from physXAI.preprocessing.training_data import TrainingDataMultiStep


def get_parent_working_directory() -> str:
    """
    Finds the root directory of the Git repository that contains the current working directory.

    This function is useful for locating project-relative paths when the script
    might be run from different subdirectories within a Git project.

    Returns:
        str: The absolute path to the root of the Git working tree if found.
             Returns an empty string if not in a Git repository or if an error occurs.
    """

    try:
        repo = git.Repo(search_parent_directories=True)
        git_root = repo.working_tree_dir
        return git_root
    except git.InvalidGitRepositoryError:  # pragma: no cover
        print(f"Error: Cannot find git root directory.")  # pragma: no cover
        return ''  # pragma: no cover
    except Exception as e:  # pragma: no cover
        print(f"Error: An unexpected error occurred when searching for parent directory: {e}")  # pragma: no cover
        return ''  # pragma: no cover


def get_full_path(path: str, raise_error=True) -> str:
    """
    Resolves a given path to an absolute path.
    If the path is relative, it first checks relative to the current working directory.
    If not found, it attempts to resolve it relative to the Git project's root directory.

    Args:
        path (str): The path string to resolve (can be absolute or relative).
        raise_error (bool, optional): If True (default), raises a FileNotFoundError
                                      if the path cannot be resolved. If False,
                                      returns the constructed path even if it doesn't exist.

    Returns:
        str: The resolved absolute path. If `raise_error` is False and the path
             is not found, it returns the last attempted path construction.

    Raises:
        FileNotFoundError: If `raise_error` is True and the path cannot be found.
    """

    if os.path.exists(path):
        return path
    parent = get_parent_working_directory()
    path = os.path.join(parent, path)
    if os.path.exists(path):
        return path
    elif raise_error:
        raise FileNotFoundError(f'Path "{path}" does not exist.')
    else:
        return path


def create_full_path(path: str) -> str:
    """
    Ensures that the directory structure for a given file path exists, creating
    it if necessary. Returns the absolute version of the input path.

    Args:
        path (str): The file path for which the directory structure should be created.
                    This can be a path to a file or just a directory.

    Returns:
        str: The absolute path, with its directory structure ensured to exist.

    Raises:
        OSError: If `os.makedirs` fails for reasons other than the directory already existing.
    """

    directory = os.path.dirname(path)
    file = os.path.basename(path)

    directory = get_full_path(directory, raise_error=False)
    path = os.path.join(directory, file)

    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            raise e

    return path


class Logger:

    save_name_preprocessing: str = 'preprocessing_config.json'
    save_name_model_config: str = 'model_config.json'
    save_name_constructed: str = 'constructed_config.json'
    save_name_training_data_multi_step: str = 'training_data'
    save_name_training_data_multi_step_format: str = 'zip'
    save_name_training_data_json: str = 'training_data.json'
    base_path = 'stored_data'
    save_name_model: str = 'model'
    save_name_model_online_learning: str = 'model_ol'

    _logger = None
    _override = False

    @staticmethod
    def override_question(path: str):  # pragma: no cover
        if os.path.exists(path) and not Logger._override:
            try:
                user_input = input(f"Path {path} already exists. Do you want to override it (y/n)?").strip().lower()
                if user_input in ['y', 'yes', 'j', 'ja', 'true', '1']:
                    shutil.rmtree(path)
                else:
                    raise OSError(f"Path {path} already exists.")
            except OSError as e:
                raise e

    @staticmethod
    def already_exists_question(path: str):  # pragma: no cover
        if os.path.exists(path) and not Logger._override:
            try:
                user_input = input(f"Path {path} already exists. Do you want to proceed (y/n)?").strip().lower()
                if user_input in ['y', 'yes', 'j', 'ja', 'true', '1']:
                    return
                else:
                    raise OSError(f"Path {path} already exists.")
            except OSError as e:
                raise e

    @staticmethod
    def setup_logger(folder_name: str = None, override: bool = False, base_path: str = None):
        if base_path is None:
            base_path = Logger.base_path
        if folder_name is None:
            folder_name = datetime.now().strftime("%d.%m.%y %H:%M:%S")
            folder_name = os.path.join(base_path, folder_name)
        else:
            folder_name = os.path.join(base_path, folder_name)
        path = get_full_path(folder_name, raise_error=False)
        if not override and os.path.exists(path):
            Logger.already_exists_question(path)
        create_full_path(path)

        Logger._logger = path
        Logger._override = override

    @staticmethod
    def log_setup(preprocessing=None, model=None, save_name_preprocessing=None, save_name_model=None,
                  save_name_constructed=None):
        if Logger._logger is None:
            Logger.setup_logger()

        if preprocessing is not None:
            try:
                preprocessing_dict = preprocessing.get_config()
            except AttributeError:  # pragma: no cover
                raise AttributeError('Error: Preprocessing object has no attribute "get_config()".')  # pragma: no cover
            if save_name_preprocessing is None:
                save_name_preprocessing = Logger.save_name_preprocessing
            path = os.path.join(Logger._logger, save_name_preprocessing)
            path = create_full_path(path)
            Logger.override_question(path)
            with open(path, "w") as f:
                json.dump(preprocessing_dict, f, indent=4)

            constructed_config = FeatureConstruction.get_config()
            if len(constructed_config) > 0:
                if save_name_constructed is None:
                    save_name_constructed = Logger.save_name_constructed
                path = os.path.join(Logger._logger, save_name_constructed)
                path = create_full_path(path)
                Logger.override_question(path)
                with open(path, "w") as f:
                    json.dump(constructed_config, f, indent=4)

            FeatureConstruction.reset()

        if model is not None:
            try:
                model_dict = model.get_config()
            except AttributeError:  # pragma: no cover
                raise AttributeError('Error: Model object has no attribute "get_config()".')  # pragma: no cover
            if save_name_model is None:
                save_name_model = Logger.save_name_model_config
            path = os.path.join(Logger._logger, save_name_model)
            path = create_full_path(path)
            Logger.override_question(path)
            with open(path, "w") as f:
                json.dump(model_dict, f, indent=4)

    @staticmethod
    def save_training_data(training_data, path: str = None):
        if Logger._logger is None:
            Logger.setup_logger()

        try:
            td_dict = training_data.get_config()
        except AttributeError:  # pragma: no cover
            raise AttributeError('Error: Training data object has no attribute "get_config()".')  # pragma: no cover

        if path is None:
            path = Logger.save_name_training_data_json
        else:
            if len(path.split('.json')) == 1:
                # join .json to path in case it is not yet included
                path = path + '.json'

        p = os.path.join(Logger._logger, path)
        p = create_full_path(p)
        Logger.override_question(p)
        with open(p, "w") as f:
            json.dump(td_dict, f, indent=4)

        if isinstance(training_data, TrainingDataMultiStep):
            training_data = copy.copy(training_data)
            training_data.train_ds = None
            training_data.val_ds = None
            training_data.test_ds = None

        p = p.split('.json')[0]
        with open(p + '.pkl', "wb") as f:
             pickle.dump(training_data, f)

    @staticmethod
    def get_model_savepath(save_name_model: str = None) -> str:
        """
        returns the path the model is saved to

        Args:
             save_name_model (str): optional name the model is saved with (string without .keras),
                                    default: Logger.save_name_model
        """
        if Logger._logger is None:
            Logger.setup_logger()
        if save_name_model is None:
            save_name_model = Logger.save_name_model

        p = os.path.join(Logger._logger, save_name_model)

        return p
