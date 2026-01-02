import os
from typing import Union
import inspect
from physXAI.feature_selection.recursive_feature_elimination import recursive_feature_elimination_pipeline
from physXAI.preprocessing.constructed import FeatureBase
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep, PreprocessingMultiStep, PreprocessingData
from physXAI.models.models import AbstractModel, SingleStepModel, MultiStepModel
from physXAI.utils.logging import Logger


class TestCase:

    def __init__(self, inputs: list[Union[str, FeatureBase]], output: Union[str, list[Union[str, FeatureBase]]],
                 data_path: str, label_width: int = None,  warmup_width: int = None):
        # TODO: add docstrings
        # label_width, warmup_width required if multistep should be used
        self.inputs = inputs
        self.output = output
        self.data_path = data_path

        # only relevant for MultiStepModels
        self.label_width = label_width
        self.warmup_width = warmup_width

    def get_preprocessing_single_step(self, **kwargs) -> PreprocessingSingleStep:
        return PreprocessingSingleStep(self.inputs, self.output, **kwargs)

    def get_preprocessing_multi_step(self, **kwargs) -> PreprocessingMultiStep:
        assert self.label_width is not None, 'label_width is required for MultiStepModel'
        assert self.warmup_width is not None, 'warmup_width is required for MultiStepModel'

        return PreprocessingMultiStep(self.inputs, self.output, self.label_width, self.warmup_width, **kwargs)

    def pipeline(self, m: AbstractModel, prep: PreprocessingData = None, plot: bool = True, save_model: bool = True,
                 online_epochs: int = -1):

        # Get the filename of executable file
        caller_frame = inspect.currentframe().f_back
        caller_filename = caller_frame.f_globals["__file__"]
        caller_filename = os.path.splitext(os.path.basename(caller_filename))[0]

        # Setup up logger for saving
        Logger.setup_logger(folder_name=caller_filename, override=True)

        # Create Training data
        if prep is None:
            if isinstance(m, SingleStepModel):
                prep = self.get_preprocessing_single_step()
            elif isinstance(m, MultiStepModel):
                prep = self.get_preprocessing_multi_step()
            else:
                raise TypeError(f'unrecognized model type: {type(m)}')

        # Process Training data
        td = prep.pipeline(self.data_path)

        # Training pipeline
        model = m.pipeline(td, plot=plot, save_model=save_model)

        if online_epochs > 0:
            assert save_model, 'save_model must be True if online learning is used'
            m.epochs = online_epochs
            model_ol = m.online_pipeline(td, load_path=os.path.join(Logger._logger, 'model.keras'), plot=plot)

        # Log setup of preprocessing and model as json
        Logger.log_setup(prep, m)
        # Log training data as pickle
        Logger.save_training_data(td)

    def recursive_feature_elimination_pipeline(self, m: SingleStepModel, prep: PreprocessingData = None, **kwargs):
        # Create Training data
        if prep is None:
            prep = self.get_preprocessing_single_step()

        return recursive_feature_elimination_pipeline(self.data_path, prep, m, **kwargs)
