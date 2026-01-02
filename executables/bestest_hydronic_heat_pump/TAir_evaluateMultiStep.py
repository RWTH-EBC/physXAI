from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep
from configuration import *

"""
Creates a linear regression to predict the air temperature using the Boptest data
The model is trained as a single step model, but evaluated as a multi step model
"""


# Create Training data
"""
The model is trained as a single step model and evaluated as a multi step model
warmup_width should be 0 as initial values cannot be processed by single step models
init_features: There are two prediction modes for multi-step: 
(0) Predict change of original variable, (1) Predict original variable
init_features should contain original variable, can be None in mode (1)
overlapping_sequence should be False to avoid duplicate labels for single step prediction
batch_size should be 1 as batches are processes differently in single step models
"""
bestest_tair.warmup_width = 0  # overwrite default set in configuration.py
prep = bestest_tair.get_preprocessing_multi_step(init_features=['reaTZon_y'], overlapping_sequences=False, batch_size=1)

# Linear Regression
m = LinearRegressionModel()

bestest_tair.pipeline(m, prep)
