from typing import Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from keras.src.activations import ALL_OBJECTS_DICT


class ClassicalANNConstruction_config(BaseModel):

    n_layers: int = Field(..., gt=0)
    n_neurons: Union[int, list[int]] = 32
    activation_function: Union[str, list[str]] = 'softplus'
    rescale_output: bool = True
    kernal_constraint: Optional[Literal["NonNeg"]] = None

    @field_validator('n_neurons')
    def validate_n_neurons(cls, v, info):
        ls = info.data.get('n_layers')
        if isinstance(v, list):
            if len(v) != ls:
                raise ValueError('List of n_neurons must have same length than n_layers')
            for i in v:
                if i <= 0:
                    raise ValueError('n_neurons must be greater than 0')
        elif v <= 0:
            raise ValueError('n_neurons must be greater than 0')
        return v

    @field_validator('activation_function')
    def validate_activation(cls, v, info):
        ls = info.data.get('n_layers')
        if isinstance(v, list):
            if len(v) != ls:
                raise ValueError('List of activation functions must have same length than n_layers')
        return v


class RBFConstruction_config(ClassicalANNConstruction_config):

    random_state: int = 42


class CMNNModelConstruction_config(ClassicalANNConstruction_config):

    monotonicities: Optional[dict[str, int]] = None
    activation_split: Optional[list[float]] = [1, 1, 1]

    @field_validator('monotonicities')
    def validate_monotonicities(cls, v):
        if v is not None:
            for i in v.values():
                if i not in [0, -1, 1]:
                    raise ValueError('monotonicities must be 0 (no monotony), 1 (positive monotony) '
                                     'or -1 (negative monotony)')
        return v

    @field_validator('activation_split')
    def validate_activation_split(cls, v):
        if v is not None:
            if len(v) != 3:
                raise ValueError('activation_split must have 3 elements')
            for i in v:
                if i < 0:
                    raise ValueError('activation_split must be greater than 0')
            if sum(v) == 0:
                raise ValueError("activation_split should contain at least one none-zero value")
        return v


class RNNModelConstruction_config(BaseModel):
    rnn_units: int = Field(32, gt=0)
    rnn_layer: Literal["RNN", "GRU", "LSTM"] = "RNN"
    init_layer: Optional[Literal["dense", "RNN", "GRU", "LSTM", "LastOutput"]] = "RNN"
    prior_layer: Optional[Literal['dense']] = None
    activation: Optional[str] = 'tanh'

    @field_validator("init_layer")
    def validate_init_layer(cls, v, info):
        if v is not None:
            if v != "dense" and v != "LastOutput":
                if v is not info.data.get('rnn_layer'):
                    raise ValueError(f"init_layer {v} should be the same as rnn_layer "
                                     f"{info.data.get('rnn_layer')} or one of (dense, LastOutput)")
        return v

    @field_validator("activation")
    def check_activation(cls, v):
        if v is not None and v not in ALL_OBJECTS_DICT:
            raise ValueError(f"activation must be one of {list(ALL_OBJECTS_DICT.keys())}")
        return v


class MonotonicRNNModelConstruction_config(RNNModelConstruction_config):

    monotonicity: Optional[dict[str, int]] = None
    dis_layer:  Optional[Literal["dense", "RNN", "GRU", "LSTM", "LastOutput", "Zero"]] = 'Zero'
    dis_units: int = Field(32, gt=0)
    dis_activation: Optional[str] = 'tanh'
    init_dis: Optional[str] = 'Zero'
    fully_connected: Optional[bool] = True

    @field_validator('monotonicity')
    def validate_monotonicity(cls, v):
        if v is not None:
            for i in v.values():
                if i not in [-1, 1]:
                    raise ValueError("monotonicity must 1 (positive monotony) "
                                     "or -1 (negative monotony). For 0 (no monotony), no need to set.")
        return v

    @field_validator("dis_activation")
    def check_dis_activation(cls, v):
        if v is not None and v not in ALL_OBJECTS_DICT:
            raise ValueError(f"dis_activation must be one of {list(ALL_OBJECTS_DICT.keys())}")
        return v

    @field_validator("init_dis")
    def validate_init_dis(cls, v, info):
        if v is not None:
            if v not in ["dense", "LastOutput", "Zero"]:
                if v is not info.data.get('dis_layer'):
                    raise ValueError(f"init_dis {v} should be the same as dis_layer "
                                     f"{info.data.get('dis_layer')} or one of (dense, LastOutput, Zero)")
        return v