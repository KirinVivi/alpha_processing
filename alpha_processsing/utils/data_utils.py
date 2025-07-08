import os
import re
import ray
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Dict, List, Optional
from pathlib import Path


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """filter out rows and columns that are all NaN or all zeros"""
    df = df.loc[~df.isna().all(axis=1)]
    df = df.loc[:, ~df.isna().all(axis=0)]
    return df.loc[:, ~(df == 0).all(axis=0)]

def clean_empty_nodes(d: dict) -> dict:
    """clean empty nodes in a nested dictionary"""
    return {
        k: v for k, v in (
            (k, clean_empty_nodes(v)) if isinstance(v, dict) else (k, v)
            for k, v in d.items()
        ) if v not in (None, {}, "")
    }

def cal_ic(ret: pd.DataFrame, alpha: pd.DataFrame, threshold: float = 0.018):
    """
    Calculate the Information Coefficient (IC) between alpha and ret.

    Args:
        ret (pd.DataFrame): Returns data with dates as index and assets as columns.
        alpha (pd.DataFrame): Alpha factor data with dates as index and assets as columns.
        threshold (float): Threshold for determining if IC is significant. Default is 0.015.

    Returns:
        tuple: A tuple containing a boolean indicating if the absolute IC is greater than threshold, and the IC value.

    Raises:
        ValueError: If DataFrames have no common columns or indices.
    """
    # Find common columns and indices more efficiently
    common_cols = ret.columns.intersection(alpha.columns)
    common_index = ret.index.intersection(alpha.index)

    # Validate that we have common data
    if common_cols.empty:
        raise ValueError("No common columns found between ret and alpha DataFrames")
    if common_index.empty:
        raise ValueError("No common indices found between ret and alpha DataFrames")

    # Align data using pandas built-in methods (more efficient than manual sorting)
    ret_aligned = ret.loc[common_index, common_cols]
    alpha_aligned = alpha.loc[common_index, common_cols]

    # Calculate IC using vectorized operations
    ic = alpha_aligned.corrwith(ret_aligned, axis=1).mean()
    rank_ic = alpha_aligned.rank(axis=1).corrwith(ret_aligned.rank(axis=1), axis=1).mean()
    # Handle NaN case
    if pd.isna(ic):
        return False, 0.0

    return (np.abs(ic) > threshold) and (np.abs(rank_ic) > threshold), ic, rank_ic

@ray.remote
def compute_corr_batch(ret: pd.DataFrame, batch: dict) -> dict:
    """
    Compute the correlation between returns and alpha factors in a batch.
    Args:
        ret (pd.DataFrame): Returns data with dates as index and assets as columns.
        batch (dict): A nested dictionary where keys are fill functions, then calculation functions, and finally smoothing functions.   
    Returns:
        dict: A nested dictionary with the same structure as the input batch, containing DataFrames that pass the IC threshold.
    """
    res_d = {}
    for fill_func, groups in batch.items():
        res_d[fill_func] = {}
        for cal_func, subgroups in groups.items():
            res_d[fill_func][cal_func] = {}
            for smooth_func, df in subgroups.items():
                ic_result = cal_ic(ret, df)
                if ic_result[0]:
                    res_d[fill_func][cal_func][smooth_func] = df, ic_result[1]
    return res_d


def save_nested_dict(data: Dict, save_dir: str, filename: Optional[str] = None) -> List[str]:
    """
    Save a nested dictionary to files in a specified directory using pathlib.Path.
    Args:
        data (Dict): The nested dictionary to save.
        {'cal_func': {'smooth_func': df}}
        save_dir (str): The directory where the files will be saved.
        filename (Optional[str]): The base filename for the saved files. If None, defaults to "result". # fill_func
    Returns:
        List[str]: A list of file paths where the data was saved.
    Raises: 
        RuntimeError: If an error occurs during saving.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_paths = []

    if filename is None:
        filename = "result"
    try:
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                   if isinstance(subvalue, (pd.DataFrame, np.ndarray)):
                        df = pd.DataFrame(subvalue) if isinstance(subvalue, np.ndarray) else subvalue
                        filepath = save_dir / f"{filename}__{key}__{subkey}.parquet"
                        df.to_parquet(filepath, engine="pyarrow", index=True, compression='zstd')
                        file_paths.append(str(filepath))
            elif isinstance(value, (pd.DataFrame, np.ndarray)):
                df = pd.DataFrame(value) if isinstance(value, np.ndarray) else value
                filepath = save_dir / f"{filename}_{key}.parquet"
                df.to_parquet(filepath, engine="pyarrow", index=True)
                file_paths.append(str(filepath))
    except Exception as e:
        raise RuntimeError(f"error happened in saving datas: {e}")
    return file_paths

def load_nested_dict(save_dir: str, batch_size: int):
    all_files = []
    for root, _, files in os.walk(save_dir):
        for f in files:
            if f.endswith(".parquet"):
                all_files.append(os.path.join(root, f))
    for i in range(0, len(all_files), batch_size):
        batch = {}
        for file_path in all_files[i:i + batch_size]:
            keys = os.path.basename(file_path).replace(".parquet", "").split("__")
            fill_func, cal_func, smooth_func = keys[0], "__".join(keys[1:-1]), keys[-1]
            batch.setdefault(fill_func, {}).setdefault(cal_func, {})[smooth_func] = pd.read_parquet(file_path, engine="pyarrow")
        yield batch
    
def extract_value_from_string(input_string: str):
    """
    Extract values from a string in the format 'func(a,b)' or 'funcN'.
    Returns a tuple of extracted values, or None if no match is found.
    """
    match = re.match(r'(\w+)\((\d+),(\d+)\)', input_string)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    match = re.match(r'(\w+)(\d+)', input_string)
    if match:
        return match.group(1), int(match.group(2))
    return None



def group_and_merge_data(
    data: pd.DataFrame,
    window_days: int,
    is_sliding: bool = False
) -> pd.DataFrame:
    """group and merge data by window size"""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data has to be a DataFrame with DatetimeIndex")

    if is_sliding:
        groups = [
            (i, data.iloc[i:i+window_days])
            for i in range(0, len(data) - window_days + 1)
        ]
    else:
        groups = [
            (i, data.iloc[i:i+window_days])
            for i in range(0, len(data), window_days)
        ]
    
    merged_dfs = []
    for group_id, group_df in groups:
        if not group_df.empty:
            multi_index = pd.MultiIndex.from_arrays(
                [group_df.index, [group_id] * len(group_df)],
                names=["datetime", "group_id"]
            )
            group_df.index = multi_index
            merged_dfs.append(group_df)
    
    if not merged_dfs:
        raise ValueError("No valid data found after grouping. Check your window size or data content.")
    
    return pd.concat(merged_dfs).sort_index()

