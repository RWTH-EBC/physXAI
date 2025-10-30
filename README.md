![E.ON EBC RWTH Aachen University](https://github.com/RWTH-EBC/physXAI/blob/main/docs/EBC_Logo.png?raw=true)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rwth-ebc.github.io/physXAI/)
![Coverage](https://raw.githubusercontent.com/RWTH-EBC/physXAI/dc3f25cbff23c06eac5344978b431a2faf27aa1c/build/reports/coverage.svg)
[![DOI:10.1016/j.buildenv.2025.113640](https://img.shields.io/badge/DOI-10.1016%2Fj.buildenv.2025.113640-227BC0)](https://doi.org/10.1016/j.buildenv.2025.113640)


<div>
<img src="https://github.com/RWTH-EBC/physXAI/blob/main/docs/physXAI.png?raw=true" height="300" alt="physXAI Logo">
</div>

# physXAI

## About The Project
The physXAI Toolbox is designed for creating physics-guided machine learning models, also known as physics-informed or physics-constrained models. <br /> 
These models are specifically intended for application in Model Predictive Control (MPC) of Building Energy Systems (BES). <br />
The toolbox aims to integrate physical knowledge into machine learning models to improve their accuracy, robustness, and interpretability for energy system optimization. <br />
<br />
If you have any questions regarding physXAI, feel free to contact us at ebc-tools@eonerc.rwth-aachen.de  

## Installation
1. Create virtual environment with Python 3.12 and activate it:
	```
	python -m venv physXAI
	physXAI\Scripts\activate.bat
	```
2. Clone pyhsXAI git repo:
	```
	git clone https://github.com/RWTH-EBC/physXAI.git
	```
3. Switch in project directory:
	```
	cd <path_to_repository>
	```
4. A) To install physXAI as a user:
	```
	pip install .
	```
 
    B) To install physXAI as a developer:
	```
	pip install -e .[dev]
	```

## Getting Started
Executable and commented examples demonstrating the use of the physXAI Toolbox can be found in the `executables` directory. <br />
These examples serve as a starting point for new projects and showcase how to configure and run the models.
New executable scripts should be added to the `executables` directory.

## Current Model Types
The physXAI currently focuses on physics-guided neural networks build with Keras and Tensorflow.
Models in this repository are categorized into two main types:

- Single-Step Models: Predicting one step ahead. Can be used recursivly in the MPC.
	- Linear Regression (sicit-learn)
	- Classical ANN
	- Radial Basis Function Network (RBF): RBF Models extrapolate to 0. Useful for residual models.
	- Residual Model: hybrid model combining a Linear Regression model with an RBF, that models the residuals of the linear regression.
	- Constrained Monotonic Neural Network (CMNN): Allows enforcing monotonicity, convex and concave constraints on input features.
	- Physics-Informed Neural Network (PINN): Allows to add physics-informed loss functions
- Multi-Step Models: Predictiing a trajectory.
	- Recurrent Neural Network: Currently allow SimpleRNN, GRU and LSTM.

## Project Structure
The project is organized into the following directories:

- `executables`: Contains executable scripts and configuration files. This is the primary location for users to run and test models.
	- The base example uses data from the BOPTEST 'Bestest Hydronic Heat Pump' test case: https://ibpsa.github.io/project1-boptest/testcases/ibpsa/testcases_ibpsa_bestest_hydronic_heat_pump/
- `data`: Storage for input data, typically in `.csv` format. Datasets required for training and evaluating models should be placed here.
- `stored_data`: This directory is used to save trained models, model parameters, and any relevant metadata generated during the modeling process.
- `physXAI`: Contains the core logic and main functionalities of the physXAI Toolbox.
	- `preprocessing`: Modules and scripts for data preprocessing tasks (e.g., cleaning, normalization, feature engineering).
	- `feature_selection`: Pipelines for (automatic) feature selection. Currently supports recursive feature elimination.
	- `models`: Modules related to model creation and definition.
		- `models.py`: Contains base model architectures and generic model components.
		- `ann`: Specific components, layers, or utilities tailored for Artificial Neural Network (ANN) based models.
	- `plotting`: Scripts and functions for generating plots and visualizations of data, model results, etc.
	- `evaluation`: Modules for evaluating model performance using various metrics.
	- `utils`: Utility functions and helper scripts used across different parts of the toolbox.
- `docs`: Contains documentation for the project.
- `build`: Contains saved test reports.
- `unittests`: Contains unittests for physXAI.

## How to contribute to the development

You are invited to contribute to the development of this toolbox.
Issues can be reported using this site's Issues section.
Furthermore, you are welcome to contribute via Pull Requests.

## Referencing physXAI

To cite physXAI, please use the following paper:

> Henkel, Patrick and Roß, Simon and Rätz, Martin and Müller, Dirk, Monotonic physics-constrained neural networks for model predictive control of building energy systems. 2025.Available at Building and Environment: https://doi.org/10.1016/j.buildenv.2025.113640

## Copyright and license
This tool is released by RWTH Aachen University, E.ON Energy Research Center, Institute for Energy Efficient Buildings and Indoor Climate <br />
and is licensed under the BSD 3 Clause License - see the [LICENSE](https://github.com/RWTH-EBC/physXAI/blob/main/LICENSE) file for details.

## Acknowledgments

We gratefully acknowledge the financial support by Federal Ministry for Economic Affairs and Climate Action (BMWK), promotional reference 03EN6022B.

<img src="https://github.com/RWTH-EBC/physXAI/blob/main/docs/BMWK_logo.png?raw=true" alt="BMWK" width="200"/>

## Contact
Patrick Henkel <br />
ebc-tools@eonerc.rwth-aachen.de <br />
https://www.ebc.eonerc.rwth-aachen.de/cms/~dmzz/E-ON-ERC-EBC/