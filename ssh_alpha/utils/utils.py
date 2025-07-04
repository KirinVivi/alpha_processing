import code
import pandas as pd
import numpy as np
from typing import Union, Dict, List, final
from pathlib import Path
import logging
import operator
from functools import reduce
from pyparsing import C
import yaml
import polars as pl
from datetime import date, timedelta
from path_set import alp_path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(table, key, freq):
    """
    Fetch data from the database based on the provided table, key, and frequency.
    Args:
        table (str): Name of the database table.
        key (str): Key to filter the data.
        freq (str): Frequency of the data, e.g., 'daily', '15', '30'.
    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """

    # Placeholder for actual database fetching logic
    # This should be replaced with actual code to fetch data from a database
    return pd.DataFrame()  # Return an empty DataFrame for now

def load_data(config: dict) -> dict:
    """
    Load data from HDF files based on the configuration.
    Args:
        config (dict): configuration dictionary containing data paths and keys.
    Returns:
        dict: a dictionary where keys are data names and values are DataFrames.
    """
    data_dict = {}
    for key, params in config.items():
        table = params["table"]
        para_freq = params["freq"]
        df = get_data( key, table, para_freq)
        data_dict[key] = df
    return data_dict

from typing import Tuple

def get_common_indcol(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.Index, pd.Index]:
    """
    Get common index and columns from multiple DataFrames.
    Args:
        data_dict (Dict[str, pd.DataFrame]): dictionary of DataFrames
    Returns:
        pd.Index: common index
        pd.Index: common columns
    """
    common_index = data_dict[next(iter(data_dict))].index
    common_columns = data_dict[next(iter(data_dict))].columns

    for df in data_dict.values():
        common_index = common_index.intersection(df.index)
        common_columns = common_columns.intersection(df.columns)

    return common_index, common_columns

def save_factor(factor: pd.DataFrame, save_path: Path) -> None:

    try:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        factor.to_parquet(save_path, engine="pyarrow", index=True, compression='zstd')
        logger.info(f"already saved {save_path}")
    except Exception as e:
        logger.error(f"save failed: {str(e)}")
        raise
# ---------------------------
def run_factor_pipeline(config_path: Path, calculate_func: callable) -> Dict[str, Union[pd.DataFrame, dict, str]]:
    """
    main function to run the factor calculation pipeline.
    Args:
        config_path(Path): Path to the configuration file (YAML format).
        calculate_func (callable): function to calculate the factor based on loaded data.
    Returns:
        Dict: includes factor data, configuration, and save path.
        example:
            {
                'factor_data': pd.DataFrame,  # calculated factor data
                'config': dict,             # configuration used for calculation
                'save_path': str            # save path of the factor data
            }
    """
    if calculate_func is None:
        raise ValueError("calculate_func must be provided and cannot be None.")
    # load configuration
    with open(config.yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"starting factor calculation: {config['factor_name']}")

    # load data from a real data source
    load_data_config = config.get('data_config', {})
    if not load_data_config:
        logger.error("data_config is empty in the configuration file.")
        raise ValueError("data_config is empty in the configuration file.") 
    data_dict = load_data(load_data_config)
    # get common index and columns
    common_index, common_columns = get_common_indcol(data_dict)
    if common_index.empty or common_columns.empty:
        logger.error("No common index or columns found in the data.")
        raise ValueError("No common index or columns found in the data.")
    # filter data to common index and columns
    sample_data = {key: df.loc[common_index, common_columns] for key, df in data_dict.items()}
    if not sample_data:
        logger.error("No data available after filtering for common index and columns.")
    # set up parameters for factor calculation
    lookback_period, delay_period = 0, 0
    if 'params' in config:
        params = config['params']
        if 'lookback_period' in params:
            logger.info(f"lookback_period is {params['lookback_period']}")
            window = params['lookback_period']
            lookback_period = int(window) if isinstance(window, int) and window > 0 else 0
        if 'delay_period' in params:
            logger.info(f"delay_period is {params['delay_period']}")
            delay = params['delay_period']
            delay_period = int(delay) if isinstance(delay, int) and delay >= 0 else 0

    logger.info(f"lookback_period: {lookback_period}, delay_period: {delay_period}")
   
    factor_params = {
        'lookback_period': lookback_period, 
        'delay_period': delay_period,
    }
    factor_params.update(config.get('params', {}))  # merge with additional params if any   
    # calculate factor
    factor = calculate_func(sample_data, **factor_params)
    if factor is None:
        factor = pd.DataFrame(index=common_index, columns=common_columns)  # placeholder

    # save factor result
    output_path = Path(config['output_dir']) / f"{config['factor_name']}.parquet"
    save_factor(factor, output_path)

    return {
        'factor_data': factor,
        'config': config,
        'save_path': str(output_path)
    }



def panel_safe_arithmetic(
    data_dict: dict,
    vars: List[str],
    operation: str = 'add',  # or 'sub'
    fill_value: Union[float, pd.DataFrame, dict] = 0
) -> pd.DataFrame:
    """
    Safely perform addition or subtraction on multiple DataFrames, intelligently handling NaN values (optimized version).

    Optimizations:
    1. **Fast path**: Provides a dedicated fast path for the most common scalar `fill_value` (such as 0),
       directly leveraging pandas' built-in `add`/`sub` methods with the `fill_value` parameter for higher performance.
    2. **Code simplification**: Simplifies the general path for handling complex `fill_value` (dict, DataFrame),
       using .loc assignment to avoid potential warnings and make the logic clearer.
    3. **Improved readability**: Uses `operator` and `functools.reduce` to clarify the arithmetic intent,
       and clearly separates the two processing logics (fast/general).
    Args:
        data_dict (dict): A dictionary where keys are variable names and values are DataFrames.
        vars (List[str]): List of variable names to be processed.
        operation (str): The arithmetic operation to perform ('add' or 'sub').
        fill_value (Union[float, pd.DataFrame, dict]): Value to fill NaN with during the operation.
    Returns:
        pd.DataFrame: Resulting DataFrame after performing the specified operation on the input DataFrames.
    """
    if not vars:
        return pd.DataFrame()

    # 将操作字符串映射到实际的 operator 函数
    op_map = {'add': operator.add, 'sub': operator.sub}
    op_func = op_map.get(operation)
    if op_func is None:
        raise ValueError("operation 必须是 'add' 或 'sub'")

    # --- 快速路径: 当 fill_value 是一个简单的数字时 ---
    if isinstance(fill_value, (int, float)):
        df_iterator = (data_dict[name] for name in vars)
        # 以第一个 DataFrame 为基础进行累积计算
        result = next(df_iterator).copy()

        # 直接使用 DataFrame 内置方法进行高效运算，这会自动处理 NaN
        # 例如 a.add(b, fill_value=0) 会将 a 和 b 中的 NaN 视为 0 来相加
        for df in df_iterator:
            result = op_func(result, df.fillna(fill_value))
        
        # 初始的 result 中可能含有 NaN，最后需要填充
        if operation == 'add':
            return result.fillna(fill_value)
        else: # 减法需要特殊处理初始的 NaN
             # 如果 result 的某个值是 NaN, fill_value - df1 - df2...
            initial_nan_mask = result.isna() & ~df.isna()
            result[initial_nan_mask] = fill_value
            return result.fillna(0) # 剩下的NaN是全为NaN的位置

    # --- 通用路径: 当 fill_value 是 dict 或 DataFrame 时 ---
    dfs = [data_dict[name] for name in vars]
    
    # 核心逻辑：一个位置只要有一个非 NaN 值，就参与计算
    not_all_nan = ~np.logical_and.reduce([df.isna() for df in dfs])

    filled_dfs = []
    for i, name in enumerate(vars):
        df = dfs[i]
        
        # 确定当前 DataFrame 的填充值
        if isinstance(fill_value, dict):
            current_fill = fill_value.get(name, 0)
        else:  # fill_value 是一个 DataFrame
            current_fill = fill_value

        filled_df = df.copy()
        
        # 定位需要被填充的 NaN (自身是 NaN，但有其他 df 在该位置有值)
        fill_mask = df.isna() & not_all_nan

        # 使用 .loc 和 fill_mask 进行精确、安全地填充
        if isinstance(current_fill, pd.DataFrame):
            # 当填充值也是 DataFrame 时，需要对齐填充
            filled_df.loc[fill_mask] = current_fill[fill_mask]
        else:
            # 标量填充
            filled_df.loc[fill_mask] = current_fill
        
        filled_dfs.append(filled_df)

    # 使用 reduce 执行连续的算术操作
    return reduce(op_func, filled_dfs)




def rolling_avg_at_day(
        df: pl.DataFrame,
        datetime_col: str,
        window_days: int
        ) -> pl.DataFrame:
    """
    Calculate the rolling average of factors at each time of day across multiple codes.
    Args:
        df (pl.DataFrame): DataFrame containing datetime and factor columns.
        datetime_col (str): Name of the column containing datetime values.
        window_days (int): Number of days for the rolling average window.
    Returns:
        pl.DataFrame: DataFrame with rolling averages of factors at each time of day.
    1. Reshape the DataFrame to long format using melt.
    2. Extract the time of day from the datetime column.
    3. Group by code and time of day, aggregating to calculate the rolling mean.
    4. Explode the DataFrame to have one row per code and time of day.
    5. Pivot the DataFrame to have codes as columns and rolling means as values.
    6. Sort the final DataFrame by datetime.
    """
    df_long = df.melt(
        id_vars = datetime_col,
        variable_name = "code",
        value_name = "factor"
    )
    df_long_with_time = df_long.with_columns(
        pl.col(datetime_col).dt.time().alias("time_of_day")

    )
    result = (
        df_long_with_time.group_by(["code", "time_of_day"])
        .agg(
            index=pl.col(datetime_col),
            factor=pl.col("factor"),
            rolling_mean=pl.col("factor")
                .sort_by(datetime_col)
                .rolling_mean(window_days, min_periods=1)
        )
        .explode([datetime_col, "factor", "rolling_mean"])
        .sort(["code", datetime_col])
    )
    final_wide = result.pivot(
        index=datetime_col,
        columns="code",
        values="rolling_mean"
    ).sort(datetime_col) 
      # Convert to pandas DataFrame if needed   
    return final_wide.to_pandas().set_index(datetime_col)




def shift_by_days(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
    n_days: int = 1
) -> pl.DataFrame:
    """
    Shift values in a wide DataFrame by a specified number of days at each time of day for each code.
    Args:           
        df (pl.DataFrame): Wide DataFrame with datetime and multiple code columns.
        datetime_col (str): Name of the column containing datetime values.
        n_days (int): Number of days to shift the values.
    Returns:
        pl.DataFrame: DataFrame with shifted values, where each code's values are shifted by n_days.
    1. Melt the DataFrame to long format, creating a column for codes and another for values.
    2. Add
        a 'time_of_day' column to group by time of day.
    3. Sort the DataFrame by code, time of day, and datetime to prepare for shifting.   
    4. Use the `shift` method to shift values by n_days within each code and time of day group.
    5. Pivot the DataFrame back to wide format, restoring the original structure with shifted values.
    6. Optionally, rename the columns to reflect the shift operation.   
    """
    # 1. Melt: 将宽表转换为长表 (datetime, code, value)
    df_long = df.melt(
        id_vars=datetime_col,
        variable_name="code",
        value_name="value"
    )

    # 2. 添加 'time_of_day' 列用于分组
    df_long_with_time = df_long.with_columns(
        pl.col(datetime_col).dt.time().alias("time_of_day")
    )

    # 3. 按分组键和时间排序（对于shift/rolling等窗口函数至关重要）
    sorted_long_df = df_long_with_time.sort(["code", "time_of_day", datetime_col])

    # 4. 核心计算：使用 shift().over()
    #    - shift(n_days): 将数据向下平移 n_days 个位置
    #    - over([...]): 指定在哪个分组内独立进行平移
    result_long = sorted_long_df.with_columns(
        pl.col("value").shift(periods=n_days).over(["code", "time_of_day"]).alias("value_shifted")
    )
    # 5. Pivot: 将计算结果从长表恢复为宽表
    final_wide = result_long.pivot(
        values="value_shifted",
        index=datetime_col,
        columns="code"
    ).sort(datetime_col)
    
    return final_wide.to_pandas().set_index(datetime_col)   