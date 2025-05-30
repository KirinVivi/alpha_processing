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

def cal_ic(ret: pd.DataFrame, alpha: pd.DataFrame, window=20):
    """
    Calculate the Information Coefficient (IC) between alpha and ret.
    Args:
        ret (pd.DataFrame): Returns data with dates as index and assets as columns.
        alpha (pd.DataFrame): Alpha factor data with dates as index and assets as columns.
        window (int): Rolling window size for calculating IC.
    Returns:
        tuple: A tuple containing a boolean indicating if the absolute IC is greater than 0.015, and the IC series.
    """
    alpha = alpha[alpha.index.isin(ret.index)]
    ret = ret[ret.index.isin(alpha.index)]
    ic = alpha.rolling(window=window).corr(ret).mean()
    return np.abs(ic) > 0.015, ic

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
                    res_d[fill_func][cal_func][smooth_func] = df
    return res_d


def save_nested_dict(data: Dict, save_dir: str, filename: Optional[str] = None) -> List[str]:
    """
    Save a nested dictionary to files in a specified directory using pathlib.Path.
    Args:
        data (Dict): The nested dictionary to save.
        save_dir (str): The directory where the files will be saved.
        filename (Optional[str]): The base filename for the saved files. If None, defaults to "result".
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
                        filepath = save_dir / f"{filename}_{key}_{subkey}.parquet"
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
            keys = os.path.basename(file_path).replace(".parquet", "").split("_")
            fill_func, cal_func, smooth_func = keys[0], "_".join(keys[1:-1]), keys[-1]
            batch.setdefault(fill_func, {}).setdefault(cal_func, {})[smooth_func] = pd.read_parquet(file_path, engine="pyarrow")
        yield batch

def extract_values_from_string(s: str):
    if isinstance(s, str) and "[" in s:
        match = re.match(r"(\w+)\[([\d\s,]+)\]", s)
        if match:
            func_name, params = match.groups()
            params = eval(f"[{params}]")
            return func_name, params
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

def select_robust_fill_methods(
    corr_results: Dict[str, Dict[str, float]],
    corr_threshold: float = 0.9,
    std_threshold: float = 0.1,
    min_valid_ratio: float = 0.8,
    max_methods: int = 3
) -> List[str]:
    """筛选鲁棒填充方法"""
    robust_methods = []
    total_cols = max([stats["n_valid"] for stats in corr_results.values()])
    
    for method, stats in corr_results.items():
        if (stats["mean_corr"] >= corr_threshold and
            stats["std_corr"] <= std_threshold and
            stats["n_valid"] / total_cols >= min_valid_ratio):
            robust_methods.append(method)
    
    return robust_methods[:max_methods] if robust_methods else list(corr_results.keys())[:max_methods]

