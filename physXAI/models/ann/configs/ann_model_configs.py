from typing import Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ClassicalANNConstruction_config(BaseModel):

    n_layers: int = Field(..., ge=0)
    n_neurons: Union[int, list[int]] = 32
    activation_function: Union[str, list[str]] = 'softplus'
    rescale_output: bool = True
    normalize: bool = True
    n_features: Optional[int] = None

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

    n_layers: int = Field(..., ge=1, le=1)
    random_state: int = 42
    rescale_mean: Optional[float] = Field(
        None, description="Mean value for z-score normalization of outputs"
    )
    rescale_sigma: Optional[float] = Field(
        None, description="Standard deviation for z-score normalization of outputs"
    )


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
    init_layer: Optional[Literal["dense", "RNN", "GRU", "LSTM"]] = "RNN"

    @field_validator("init_layer")
    def validate_init_layer(cls, v, info):
        if v is not None:
            if v != "dense":
                if v is not info.data.get('rnn_layer'):
                    raise ValueError(f"init_layer {v} should be the same as rnn_layer "
                                     f"{info.data.get('rnn_layer')} or dense")
        return v
    

class RC1R1CConstruction_config(ClassicalANNConstruction_config):
    """
    
    """
    
    predict_delta: bool = True
    t_air_column: Union[str, int]
    trainable_rc: bool = False
    use_internal_gains: bool = False

    physics_loss_weight: float = Field(1.0, ge=0)

    rc_kwargs: dict = Field(default_factory=dict)
    

class RC2R2CPhysNetConstruction_config(BaseModel):
    """
    
    """
    predict_delta: bool = True
    t_air_column: Union[str, int]
    trainable_rc: bool = False
    use_internal_gains: bool = False

    encoder_features: list[Union[str, int]]
    encoder_layers: int = Field(2, ge=1)
    encoder_neurons: Union[int, list[int]] = 24

    dynamic_features: list[Union[str, int]]
    dynamic_layers: int = Field(1, ge=1)
    dynamic_neurons: Union[int, list[int]] = 128
    
    activation_function: Union[str, list[str]] = 'softplus'
    rescale_output: bool = True
    normalize: bool = True
    n_features: Optional[int] = None

    physics_loss_weight: float = Field(1.0, ge=0)

    rc_kwargs: dict = Field(default_factory=dict)

    @field_validator('encoder_neurons')
    def validate_encoder_neurons(cls, v, info):
        ls = info.data.get('encoder_layers')
        if isinstance(v, list):
            if len(v) != ls:
                raise ValueError("List of encoder_neurons must have same length as encoder_layers")
            for i in v:
                if i <= 0:
                    raise ValueError("encoder_neurons must be greater than 0")
        elif v <= 0:
            raise ValueError("encoder_neurons must be greater than 0")
        return v
    
    @field_validator('dynamic_neurons')
    def validate_dynamic_neurons(cls, v, info):
        ls = info.data.get('dynamic_layers')
        if isinstance(v, list):
            if len(v) != ls:
                raise ValueError("List of dynamic_neurons must have same length as dynamic_layers")
            for i in v:
                if i <= 0:
                    raise ValueError("dynamic_neurons must be greater than 0")
        elif v <= 0:
            raise ValueError("dynamic_neurons must be greater than 0")
        return v
    

class RC2R2CGokhalePhysNetConstruction_config(BaseModel):
    """
    
    """
    predict_delta: bool = True

    t_air_column: Union[str, int]
    trainable_rc: bool = False
    use_internal_gains: bool = False

    encoder_features: list[Union[str, int]]
    encoder_layers: int = Field(2, ge=1)
    encoder_neurons: Union[int, list[int]] = 24

    dynamic_features: list[Union[str, int]]
    dynamic_layers: int = Field(1, ge=1)
    dynamic_neurons: Union[int, list[int]] = 128
    
    activation_function: Union[str, list[str]] = 'softplus'
    rescale_output: bool = True
    normalize: bool = True
    n_features: Optional[int] = None

    physics_loss_weight: float = Field(1.0, ge=0)

    rc_kwargs: dict = Field(default_factory=dict)

    @field_validator('encoder_neurons')
    def validate_encoder_neurons(cls, v, info):
        ls = info.data.get('encoder_layers')
        if isinstance(v, list):
            if len(v) != ls:
                raise ValueError("List of encoder_neurons must have same length as encoder_layers")
            for i in v:
                if i <= 0:
                    raise ValueError("encoder_neurons must be greater than 0")
        elif v <= 0:
            raise ValueError("encoder_neurons must be greater than 0")
        return v
    
    @field_validator('dynamic_neurons')
    def validate_dynamic_neurons(cls, v, info):
        ls = info.data.get('dynamic_layers')
        if isinstance(v, list):
            if len(v) != ls:
                raise ValueError("List of dynamic_neurons must have same length as dynamic_layers")
            for i in v:
                if i <= 0:
                    raise ValueError("dynamic_neurons must be greater than 0")
        elif v <= 0:
            raise ValueError("dynamic_neurons must be greater than 0")
        return v
    
    @field_validator('activation_function')
    def validate_activation_function(cls, v, info):
        encoder_layers = info.data.get('encoder_layers')
        dynamic_layers = info.data.get('dynamic_layers')

        if encoder_layers is None or dynamic_layers is None:
            return v
        
        expected_lenght = encoder_layers + dynamic_layers

        if isinstance(v, list):
            if len(v) != expected_lenght:
                raise ValueError("List of activation_function must have the same length as encoder_layers dynamic_layers")
            
        return v
    

class RC2R2CGokhalePhysNetWallDynamicsConstruction_config(RC2R2CGokhalePhysNetConstruction_config):
    """
    
    """
    wall_dynamics_loss_weight: float = Field(1.0, ge=0)