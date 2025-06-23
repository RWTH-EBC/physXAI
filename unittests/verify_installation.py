import sys
from pathlib import Path
import git


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
        path = Path(__file__)
        repo = git.Repo(path, search_parent_directories=True)
        git_root = repo.working_tree_dir
        return git_root
    except git.InvalidGitRepositoryError:
        print(f"Error: Cannot find git root directory.")
        return ''
    except Exception as e:
        print(f"Error: An unexpected error occurred when searching for parent directory: {e}")
        return ''
parent = get_parent_working_directory()
sys.path.insert(0,parent)


######################################################################################################################

from core_functions.preprocessing.preprossesing import PreprocessingSingleStep
from core_functions.models.models import LinearRegressionModel
from core_functions.utils.logging import Logger


"""
Creates standard model to predict the power of the heat pump using the Boptest data
"""

# Setup up logger for saving
Logger.setup_logger(folder_name='unittests\\verify_installation', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Can include constructed features
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']
# Output feature
output = 'reaPHeaPum_y'

# Create Training data
prep = PreprocessingSingleStep(inputs, output)
# Process Training data
td = prep.pipeline(file_path)

# Linear Regression
m = LinearRegressionModel()

# Training pipeline
model = m.pipeline(td, plot=False, save_model=False)
