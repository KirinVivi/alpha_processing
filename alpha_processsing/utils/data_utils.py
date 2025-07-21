import os
import re
import ray
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Optional: Configure logging for clear feedback
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _traverse_and_save(node: Any, key_path: List[str], save_dir_path: Path, base_filename: str, saved_files: List[str]):
    if isinstance(node, (pd.DataFrame, np.ndarray)):
        df_to_save = pd.DataFrame(node) if isinstance(node, np.ndarray) else node
        full_key_str = "__".join(key_path)
        output_filename = f"{base_filename}__{full_key_str}.parquet"
        filepath = save_dir_path / output_filename
        df_to_save.to_parquet(filepath, engine="pyarrow", index=True, compression='zstd')
        saved_files.append(str(filepath))
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _traverse_and_save(value, key_path + [key])

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
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []
    base_filename = filename if filename is not None else "result"
    try:
        _traverse_and_save(data, [], save_dir_path, base_filename, saved_files)
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving data: {e}") from e
    return saved_files

def delete_drop_mid_data(save_dir: str, drop_list: list):
    dir_path = Path(save_dir)

    # The function won't crash if the directory doesn't exist.
    if not dir_path.is_dir():
        logging.warning(f"Directory '{dir_path}' not found. Nothing to delete.")
        return
    for filename in drop_list:
        file_path = dir_path / f"{filename}.parquet"
        try:
            file_path.unlink(missing_ok=True)
        except OSError as e:
            # This catches other errors, like permission issues, without crashing.
            logging.error(f"Failed to delete {file_path}: {e}")


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
    match = re.match(r'(\w+)_(\d+)', input_string)
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


def restructure_results(flat_dict: Dict[Tuple[Any, str], pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    將扁平的結果字典重組為嵌套字典。
    
    輸入範例 (flat_dict):
    {
        (('ts_rank_60',), 'smooth_A'): pd.DataFrame(...),
        (('ts_rank_60',), 'smooth_B'): pd.DataFrame(...),
        (('filt_top_30',), 'smooth_A'): pd.DataFrame(...)
    }

    輸出範例 (nested_dict):
    {
        'ts_rank_60': {
            'smooth_A': pd.DataFrame(...),
            'smooth_B': pd.DataFrame(...)
        },
        'filt_top_30': {
            'smooth_A': pd.DataFrame(...)
        }
    }
    """
    # defaultdict(dict) 讓我們在訪問一個新 key 時，會自動為它創建一個空字典
    nested_dict = defaultdict(dict)
    
    for full_key, df in flat_dict.items():
        l1_tuple, l2_func = full_key
        
        # 為了讓 key 更美觀，我們將 L1 的元組 key 轉換為用 '__' 連接的字串
        # 例如 ('ts_rank_60', 'filt_top_30') -> 'ts_rank_60__filt_top_30'
        # 如果只有一個元素，如 ('ts_rank_60',)，則變為 'ts_rank_60'
        cal_func_key = "__".join(l1_tuple)
        
        # 將 smooth_func 作為內層字典的 key
        smooth_func_key = l2_func
        
        # 構建嵌套結構
        nested_dict[cal_func_key][smooth_func_key] = df
        
    # 將 defaultdict 轉換回普通的 dict，以便輸出
    return dict(nested_dict)

def interpolator_arr(arr_input: tuple[torch.Tensor, np.ndarray], device: str = 'cpu') -> np.ndarray:
    """
    Interpolates a PyTorch tensor to handle NaNs along axis 0 (column-wise),
    using vectorized linear interpolation with extrapolation.

    Args:
        arr_input (Tuple[torch.Tensor, np.ndarray]): Input tuple containing a PyTorch tensor and a NumPy array,
                                                      shape (num_samples, num_features).
        device (str): Device to run computation on ('cpu' or 'cuda').

    Returns:
        np.ndarray: Interpolated tensor with NaNs filled, same shape as input.
                    Returns all NaNs if there are less than 2 valid points in a column.
    """
    if isinstance(arr_input, np.ndarray):
        # If input is a NumPy array, convert to PyTorch tensor
        arr = torch.from_numpy(arr_input).float().to(device)
    original_ndim = arr.ndim
    # Handle all NaNs in the entire input (early exit)
    if torch.all(torch.isnan(arr)):
        return torch.full(arr.shape, float('nan'), device=device)

    # If 1D, temporarily make it 2D (num_samples, 1) for consistent 2D processing
    if original_ndim == 1:
        arr = arr.unsqueeze(1) # Shape (N, 1)

    num_samples, num_features = arr.shape

    # Mask for valid (non-NaN) values
    valid_mask = ~torch.isnan(arr) # Shape (N, M)

    # Check if less than 2 valid points in *any* column
    valid_counts_per_col = valid_mask.sum(dim=0) # Shape (M,)
    cannot_interpolate_cols_mask = (valid_counts_per_col < 2) # Shape (M,)

    # All possible indices along the sample dimension (N,)
    all_indices_samples = torch.arange(num_samples, dtype=torch.float32, device=device).unsqueeze(1) # Shape (N, 1)
    prev_valid_idx = torch.zeros_like(arr, dtype=torch.long, device=device)
    for i in range(1, num_samples):
        prev_valid_idx[i] = torch.where(valid_mask[i], i, prev_valid_idx[i-1])
    # Handle leading NaNs by setting their prev_valid_idx to the index of the first actual valid element in that column
    first_valid_row_idx = torch.argmax(valid_mask.long(), dim=0) # Index of first True (0 if no True)
    # If a column has no True, argmax returns 0. Check cannot_interpolate_cols_mask.
    
    # Ensure prev_valid_idx points to valid data for leading NaNs
    # For columns with all NaNs, first_valid_row_idx will be 0, but we'll set them to NaN later.
    for col_idx in range(num_features):
        if valid_counts_per_col[col_idx] > 0: # Only if column has at least one valid point
            first_idx_in_col = first_valid_row_idx[col_idx]
            prev_valid_idx[0:first_idx_in_col, col_idx] = first_idx_in_col
        else:
            prev_valid_idx[:, col_idx] = 0 # Dummy value, will be NaN later

    # Next valid index (each element gets index of first non-nan element on or after it)
    # Simulate bfill for indices by flipping, ffill, then flipping back
    next_valid_idx_temp = torch.zeros_like(arr, dtype=torch.long, device=device)
    flipped_valid_mask = valid_mask.flip(dims=[0])
    flipped_indices = torch.arange(num_samples, dtype=torch.long, device=device).flip(dims=[0])
    
    for i in range(1, num_samples):
        next_valid_idx_temp[i] = torch.where(flipped_valid_mask[i], flipped_indices[i], next_valid_idx_temp[i-1])
    next_valid_idx = next_valid_idx_temp.flip(dims=[0])

    # Ensure next_valid_idx points to valid data for trailing NaNs
    last_valid_row_idx = (num_samples - 1) - torch.argmax(valid_mask.flip(dims=[0]).long(), dim=0) # Index of last True
    for col_idx in range(num_features):
        if valid_counts_per_col[col_idx] > 0:
            last_idx_in_col = last_valid_row_idx[col_idx]
            next_valid_idx[last_idx_in_col+1:num_samples, col_idx] = last_idx_in_col
        else:
            next_valid_idx[:, col_idx] = 0 # Dummy value, will be NaN later

    # --- Get Values at Previous/Next Valid Indices ---
    # These indices are now guaranteed to be within [0, num_samples-1]
    prev_val = arr.gather(0, prev_valid_idx) # Shape (N, M)
    next_val = arr.gather(0, next_valid_idx) # Shape (N, M)
    
    # --- Calculate Distances ---
    dist_prev = all_indices_samples - prev_valid_idx.float() # Shape (N, M)
    dist_next = next_valid_idx.float() - all_indices_samples # Shape (N, M)

    # --- Calculate Denominator (for interpolation weights) ---
    denom = dist_prev + dist_next # Shape (N, M) 
    # Set to 1.0 where denom is 0 (for non-nan points; NaNs with denom=0 are handled by mask)
    denom = torch.where(denom == 0, torch.tensor(1.0, device=device), denom)

    # --- Calculate Interpolated Values ---
    interpolated_val = (prev_val * dist_next + next_val * dist_prev) / denom
    
    # --- Fill Original NaNs with Interpolated Values ---
    output_arr = torch.where(torch.isnan(arr), interpolated_val, arr)

    # --- Apply Mask for Columns with Less Than 2 Valid Points ---
    output_arr[:, cannot_interpolate_cols_mask] = torch.nan

    # If original input was 1D, squeeze back to 1D
    if original_ndim == 1:
        output_arr = output_arr.squeeze(1) # Shape (N,)
    
    return output_arr.cpu().numpy()