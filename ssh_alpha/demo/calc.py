import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

import logging
from typing import Union, Dict
from utils.utils import loop_params, panel_safe_arithmetic, rolling_avg_at_day, shift_by_days

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
def calculate_factor(
    data_dict: Dict[str, pd.DataFrame],
    lookback_period: int = 0,
    delay_period: int = 0,
    **kwargs
) -> pd.DataFrame:
    """
    calculate factor based on input data.
    Args:
        data_dict (Dict[str, pd.DataFrame]):
        lookback_period (int): lookback period for the factor calculation
        **kwargs: additional parameters for factor calculation
    Returns:
        pd.DataFrame: calculated factor data
    """
    # set variables from data_dict
    close = data_dict.get('close')
    buy_price1 = data_dict.get('buy_price1')
    sell_price1 = data_dict.get('sell_price1')
    # handle missing data and extra circumstances
    # sell_price1 > buy_price1, to avoid extra circumstances
    # find the position and fill it with close price which sell_price1 and buy_price1 are not both equal to 0
    buy_price1 = buy_price1.replace(0, np.nan)
    sell_price1 = sell_price1.replace(0, np.nan)
    # calculation logic
    # Todo: replace with actual factor calculation logic
    bid_ask_diff = panel_safe_arithmetic(
        {'sell_price1': sell_price1, 'buy_price1': buy_price1}, 
        vars=['sell_price1', 'buy_price1'], 
        operation='sub', 
        fill_value=data_dict['close']
        )
    bid_ask_diff_percent = 100*bid_ask_diff / close
    buy_overmed_volume = panel_safe_arithmetic(
        data_dict, 
        vars=['buy_volume_exlarge_order', 'buy_volume_large_order', 'buy_volume_med_order'], 
        operation='add', 
        fill_value=0
    )
    # replace zeros with NaN to avoid division by zero
    buy_overmed_volume = buy_overmed_volume.replace(0, np.nan)
    sqrt_buy_overmed_volume = np.sqrt(buy_overmed_volume)
    # and then fill NaN with median to avoid NaN in the final factor data
    median_val = sqrt_buy_overmed_volume.median()
    sqrt_buy_overmed_volume = sqrt_buy_overmed_volume.fillna(median_val)
    factor_data = bid_ask_diff_percent / sqrt_buy_overmed_volume
    # ensure the factor data is a DataFrame with the same index and columns as input data
    factor_data = pd.DataFrame(factor_data, index=close.index, columns=close.columns)
    # apply lookback and delay periods if specified
    if lookback_period > 0:
        factor_data = pl.from_pandas(factor_data)
        factor_data = rolling_avg_at_day(factor_data, "index", lookback_period)       
    if delay_period > 0:
        factor_data = pl.from_pandas(factor_data)
        factor_data = shift_by_days(factor_data, "index", delay_period)
    return factor_data
    

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    logger.info(f"Loading configuration from {config_path}")
    # run the factor calculation pipeline
    loop_params(config_path, calculate_factor)
    