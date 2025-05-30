"""
This module provides configuration validation for the FilterAlpha application using Pydantic models.
Classes:
    FilterAlphaConfig: Pydantic model that defines and validates the configuration parameters required by FilterAlpha.
Functions:
    load_config(file_path: str) -> FilterAlphaConfig:
        Loads and validates configuration from a YAML file.
    validate_config(config: Dict[str, Union[str, float, int]]) -> FilterAlphaConfig:
        Validates a configuration dictionary and returns a FilterAlphaConfig instance.
"""
from pydantic import BaseModel, Field, field_validator
import yaml
from typing import Dict, Union

class FilterAlphaConfig(BaseModel):
    """Configuration model for FilterAlpha."""
    start_date: str = Field(..., description="Start date for backtest")
    ic: float = Field(ge=0, description="Information Coefficient threshold")
    icir: float = Field(ge=0, description="Information Coefficient IR threshold")
    dd: float = Field(ge=0, description="Drawdown threshold")
    win: float = Field(ge=0, description="Win ratio threshold")
    ret: float = Field(ge=0, description="Return threshold")
    ret_l: float = Field(ge=0, description="Long return threshold")
    tvr_max: float = Field(..., description="Maximum turnover threshold")
    tvr_min: float = Field(ge=0, description="Minimum turnover threshold")
    sharpe: float = Field(ge=0, description="Sharpe ratio threshold")
    cover: float = Field(ge=0, description="Cover ratio threshold")
    out_path: str = Field(..., description="Output path for decay files")

    @field_validator('tvr_min')
    @classmethod
    def validate_tvr(cls, v, values):
        """Ensure tvr_min is not greater than tvr_max."""
        tvr_max = values.data.get('tvr_max') if hasattr(values, 'data') else values.get('tvr_max')
        if tvr_max is not None and v > tvr_max:
            raise ValueError("tvr_min cannot be greater than tvr_max")
        return v

def load_config(file_path: str) -> FilterAlphaConfig:
    """Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        FilterAlphaConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the configuration is invalid.
    """
    with open(file_path, 'r') as f:
        return FilterAlphaConfig(**yaml.safe_load(f))

def validate_config(config: Dict[str, Union[str, float, int]]) -> FilterAlphaConfig:
    """Validate configuration dictionary.

    Args:
        config (Dict[str, Union[str, float, int]]): Configuration dictionary.

    Returns:
        FilterAlphaConfig: Validated configuration object.

    Raises:
        ValueError: If the configuration is invalid.
    """
    return FilterAlphaConfig(**config)