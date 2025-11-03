# GitHub Copilot Review Agent Instructions

## Repository Overview

**physXAI** is a toolbox for creating physics-guided machine learning models, also known as physics-informed or physics-constrained models. The toolbox is specifically designed for application in Model Predictive Control (MPC) of Building Energy Systems (BES). It provides preprocessing pipelines, various neural network architectures (including classical ANNs, RBF networks, residual models, constrained monotonic networks, RNNs, and physics-informed neural networks), feature selection tools, evaluation metrics, and visualization utilities.

### Key Information
- **Language**: Python (3.9-3.12)
- **License**: BSD-3-Clause
- **Testing**: pytest (unit tests live in `unittests/`)
- **Dependencies**: Core (numpy, pandas, scikit-learn, keras, tensorflow, pydantic, plotly), Optional (dev dependencies for testing and coverage)
- **CI/CD**: GitHub Actions (runs pytest, coverage) and documentation under `docs/`

## Project Structure

```
physXAI/
├── models/                 # Core model architectures and base classes
│   ├── models.py          # AbstractModel, SingleStepModel, MultiStepModel base classes and MODEL_CLASS_REGISTRY
│   ├── ann/               # Artificial Neural Network models and components
│   │   ├── ann_design.py  # ANNModel abstract base class and concrete implementations
│   │   ├── keras_models/  # Custom Keras layers, constraints, and activations (CRITICAL)
│   │   │   └── keras_models.py  # NonNegPartial, ConcaveActivation, SaturatedActivation, etc.
│   │   ├── configs/       # Pydantic configuration models for ANN architectures
│   │   ├── model_construction/  # Model builder classes (ANN, RBF, Residual, RNN)
│   │   └── pinn/          # Physics-Informed Neural Network loss functions
├── preprocessing/          # Data preprocessing and feature engineering
│   ├── preprocessing.py   # PreprocessingData abstract base class
│   ├── training_data.py   # TrainingData, TrainingDataMultiStep, TrainingDataGeneric containers
│   └── constructed.py     # FeatureBase and FeatureConstruction for feature engineering
├── feature_selection/      # Automated feature selection pipelines
├── evaluation/             # Metrics classes for model evaluation
├── plotting/               # Visualization and plotting utilities
└── utils/                  # Logging, path utilities, and helper functions
executables/                # Runnable example scripts and use cases
unittests/                  # pytest test-suite
docs/                       # MkDocs documentation
data/                       # Input datasets (typically .csv files)
stored_data/                # Saved models and configurations
```

### Critical Architecture Components
- **Model base classes** (`physXAI/models/models.py`): `AbstractModel` defines the core model interface (generate, compile, fit, evaluate, save, load) and the `MODEL_CLASS_REGISTRY` for dynamic model instantiation. `SingleStepModel` and `MultiStepModel` provide specialized base classes. **CRITICAL**: Changes to these base classes affect all model implementations.
  
- **Model registry** (`physXAI/models/__init__.py`): **MAXIMUM IMPORTANCE** - This file imports from `keras_models.py` and controls which model classes are available in the public API. Changes here directly impact backward compatibility and all downstream code that imports models.

- **Custom Keras components** (`physXAI/models/ann/keras_models/keras_models.py`): **MAXIMUM IMPORTANCE** - Contains custom Keras serializable classes: constraints (`NonNegPartial`), activations (`ConcaveActivation`, `SaturatedActivation`, `LimitedActivation`), and custom layers. These are used throughout trained models and config files. **Changes to existing classes are typically NOT backward compatible** as they affect model serialization/deserialization. New classes can be added but should be reviewed carefully for naming conflicts and serialization compatibility.

- **ANN model implementations** (`physXAI/models/ann/ann_design.py`): Concrete implementations of ANNModel subclasses (ClassicalANN, CMNNN, RBF, ResidualModel, RNN, PINN). These orchestrate model construction, compilation, fitting, and evaluation. Changes here affect training behavior and model outputs.

- **Model construction** (`physXAI/models/ann/model_construction/`): Builder classes that generate Keras model architectures from configuration. These define layer stacks, activation patterns, and constraint application.

- **Configuration models** (`physXAI/models/ann/configs/ann_model_configs.py`): Pydantic v2 models defining configuration schemas for different ANN architectures. These form part of the public contract and are used in example configs and saved model metadata.

- **Training data containers** (`physXAI/preprocessing/training_data.py`): `TrainingDataGeneric`, `TrainingData`, and `TrainingDataMultiStep` hold datasets, predictions, metrics, and training records. These are passed throughout the model pipeline and must maintain consistent interfaces.

- **Preprocessing pipelines** (`physXAI/preprocessing/preprocessing.py`): `PreprocessingData` abstract base class and implementations handle data loading, train/val/test splitting, normalization, and feature construction.

- **Feature construction** (`physXAI/preprocessing/constructed.py`): `FeatureBase` and `FeatureConstruction` provide a framework for defining and applying feature engineering transformations.

- **Examples & CI expectations**: `executables/` contains runnable scripts (e.g., `bestest_hydronic_heat_pump/`) that demonstrate model usage and may be exercised by CI. `unittests/` contains pytest tests validating core functionality.

### Configuration Pattern
Configuration models use Pydantic v2 (e.g., `ann_model_configs.py`) with `Field()`, `field_validator` decorators. These schemas are part of the public API contract and affect:
- Saved model configurations in `stored_data/`
- Example scripts in `executables/`
- Model instantiation via `model_from_config()` and `from_config()`

Reviewers should validate schema changes carefully: prefer additive, optional fields for compatibility; when renaming/removing fields, add clear deprecation paths.


## Review Guidelines

### Core Principle: Concise, High-Impact Reviews
**Focus only on critical issues that affect:**
- Backward compatibility and breaking changes
- CI/CD pipeline failures
- Missing tests for new functionality
- Undocumented changes to public APIs

**Avoid commenting on:**
- Minor refactoring preferences
- Well-tested internal implementation details

### Primary Objectives
1. **Provide concise PR summary** (2-3 sentences max) highlighting purpose and scope
2. **Classify changes** as bug fixes, new features, refactoring, or documentation
3. **Assess backward compatibility** - flag breaking changes to public APIs
4. **Highlight CI risks** - identify changes likely to cause test/build failures
5. **Respond in English only**

### Specific Review Focus Areas

#### 1. Breaking Changes & Compatibility
- **Flag breaking changes** to: 
  - **CRITICAL**: `physXAI/models/__init__.py` - Any change to this file should be reviewed with MAXIMUM IMPORTANCE as it controls the public model API and all imports.
  - **CRITICAL**: `physXAI/models/ann/keras_models/keras_models.py` - Changes to existing custom Keras classes (constraints, activations, layers) are typically NOT backward compatible due to model serialization. Modifying class signatures, `__call__` methods, or `get_config()` breaks loading of previously saved models. New classes can be added but require careful review for naming conflicts and proper Keras serialization registration.
  - Public model base classes in `physXAI/models/models.py` (AbstractModel, SingleStepModel, MultiStepModel, MODEL_CLASS_REGISTRY).
  - Public ANN model classes in `physXAI/models/ann/ann_design.py` (ANNModel, ClassicalANN, CMNNN, RBF, etc.).
  - Configuration schemas in `physXAI/models/ann/configs/ann_model_configs.py` (Pydantic models).
  - Training data interfaces in `physXAI/preprocessing/training_data.py` (TrainingDataGeneric and subclasses).
  - Preprocessing base class in `physXAI/preprocessing/preprocessing.py`.
  - Model construction builder APIs in `physXAI/models/ann/model_construction/`.

- **Configuration schema changes**: Any modification to Pydantic models in `physXAI/models/ann/configs/` affects saved model configs in `stored_data/` and example scripts in `executables/`. Prefer additive, optional fields for compatibility; when renaming/removing fields, add a clear deprecation path and migration notes.

- **Model interface changes**: Changing method signatures, return types, or parameter semantics in base model classes (`AbstractModel`, `SingleStepModel`, `MultiStepModel`) is breaking — all model implementations depend on stable interfaces.

- **Keras serialization**: Changes to custom Keras components must maintain serialization compatibility. The `@keras.saving.register_keras_serializable()` decorator and `get_config()`/`from_config()` methods are critical for saving/loading models.

- **Training data structure changes**: Modifications to `TrainingData`, `TrainingDataMultiStep`, or `TrainingDataGeneric` attributes or methods affect all preprocessing, training, and evaluation code.

#### 2. Code Quality & Standards
- **Pydantic usage**: Ensure v2 syntax (e.g., `Field()`, `field_validator`, not v1 `validator`)
- **Type hints**: Ensure type hints are added for new functions/methods
- **Docstrings**: Check for docstrings on new public classes/methods (Google style)

#### 3. Testing Requirements
- **New features require tests**: Flag new modules, functions, or classes without corresponding test files
- **Tests location**: Tests are located in the `unittests/` folder
- **Coverage concerns**: Highlight complex logic added without test coverage

#### 4. Documentation & Examples
- **Examples location**: Examples can be found in the `executables/` folder
- **Example functionality**: Add examples for new features and functionality
- **Documentation**: Ensure new public APIs are documented in `docs/`

#### 5. CI/CD Considerations
- **Import errors**: New dependencies must be in `pyproject.toml` dependencies or optional-dependencies
- **Python version compatibility**: Code must work on Python 3.9-3.12
- **Test execution**: Changes to `unittests/` structure or execution patterns
- **Example scripts**: `executables/` should remain functional (may be tested by CI)
- **Keras/TensorFlow compatibility**: Ensure compatibility with keras and tensorflow


### Review Output Format

Provide structured, concise feedback (max 10 bullet points total):
1. **Summary**: 2-3 sentence overview of PR purpose
2. **Change Classification**: Bug fix / Feature / Refactor / Documentation / Mixed
3. **Backward Compatibility**: Compatible / Breaking (explain) / Deprecation needed
4. **CI Risk**: Low / Medium / High (explain if Medium/High)
5. **Key Issues**: Numbered list of critical concerns only (max 5 items)
6. **Suggestions**: Optional - only for significant improvements

**Brevity is essential. Skip sections with nothing critical to report.**

### What NOT to Flag
- Minor style issues
- Personal preference on implementation approach
- Overly detailed nitpicking on internal/private methods
- Changes to examples or test code (unless broken)

## Common Patterns to Recognize

- **Model base class hierarchy**: Look for `AbstractModel` → `SingleStepModel`/`MultiStepModel` → `ANNModel` → concrete implementations (ClassicalANN, CMNNN, RBF, etc.). Pay attention to method overrides and the model lifecycle (generate → compile → fit → evaluate).

- **MODEL_CLASS_REGISTRY pattern**: The registry in `models.py` enables dynamic model instantiation via `model_from_config()`. New model classes must be decorated with `@register_model` to be available for config-based instantiation.

- **Custom Keras serialization**: Custom Keras components use `@keras.saving.register_keras_serializable()` and must implement `get_config()` and optionally `from_config()`. These are critical for model persistence.

- **Pydantic configuration models**: Look for configuration classes in `physXAI/models/ann/configs/ann_model_configs.py` with `field_validator` decorators. These validate hyperparameters and enforce constraints (e.g., `n_layers > 0`, list lengths match).

- **Model construction pattern**: Model builders in `model_construction/` receive TrainingData and config objects, then programmatically construct Keras layer stacks. Watch for layer ordering, activation function application, and constraint enforcement (e.g., `NonNegPartial` for monotonicity).

- **Training data flow**: `TrainingDataGeneric` and subclasses hold datasets and flow through the pipeline: preprocessing → model.fit → predictions → metrics → plotting. Attribute names (`X_train`, `y_train`, `X_train_single`, etc.) are part of the interface contract.

- **Preprocessing pipeline pattern**: `PreprocessingData` subclasses implement `pipeline()` to load CSV, split data, normalize, construct features, and return a populated `TrainingData` object. The `shift` parameter controls time-series forecasting horizon.

- **Feature construction**: `FeatureBase` subclasses define transformations (e.g., `ExponentialFeature`, `PowerFeature`) and auto-register with `FeatureConstruction` via `__init__`. Features support arithmetic operations for derived features.

- **PINN loss functions**: Physics-Informed Neural Networks in `pinn/` use custom loss functions that combine data-driven MSE with physics-based penalties. These require careful handling of symbolic/numerical derivatives.

- **Examples & CI expectations**: `executables/` contains runnable scripts (e.g., `bestest_hydronic_heat_pump/`) that demonstrate end-to-end workflows. These may be exercised by CI and should remain functional after API changes.

- **Model save/load pattern**: Models are saved as `.keras` files (Keras format) and/or `.joblib` (scikit-learn). Configuration and metadata are saved separately as JSON. Loading requires matching Keras custom objects via the serialization registry.

- **Config injection**: Models receive hyperparameters via constructor arguments and Pydantic config objects. Validate parameter names, types, and defaults when reviewing model constructors or config classes.

---
*Instructions version: 1.0 | Target: GitHub Copilot Review Agent | Repository: physXAI*
