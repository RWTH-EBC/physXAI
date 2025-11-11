import importlib
import os
import pathlib

# Load train function from executables/agentlib_mpc/one_room_mpc/T_ann.py example
file_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'executables', 'agentlib_mpc', 'one_room_mpc', 'T_ann.py')
module_name = "T_ann"
function_name = "train_model"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

train_model = getattr(module, function_name)