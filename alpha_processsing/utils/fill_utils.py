
from bz2 import compress
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional, Union, Callable


def ffill(data: np.ndarray) -> np.ndarray:
    data = np.copy(data)
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(data.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    data[mask] = data[idx[mask], np.arange(data.shape[1])[None, :][mask]]
    return data

def bfill(data: np.ndarray) -> np.ndarray:
    data = np.copy(data)
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(data.shape[0])[:, None], data.shape[0] - 1)
    np.minimum.accumulate(idx[::-1], axis=0, out=idx[::-1])
    data[mask] = data[idx[mask], np.arange(data.shape[1])[None, :][mask]]
    return data

def fill_cross_sectional_mean(data: np.ndarray) -> np.ndarray:
    """Fill missing values with the mean of each row (cross-sectional mean)."""
    data = np.copy(data)
    row_means = np.nanmean(data, axis=1, keepdims=True)
    mask = np.isnan(data)
    data[mask] = row_means[mask]
    return data

def fill_linear_interd(data: np.ndarray) -> np.ndarray:
    """ Fill missing values using linear interpolation. """
    data = np.copy(data)
    for col in range(data.shape[1]):
        x = data[:, col]
        mask = np.isnan(x)
        if not mask.all():
            indices = np.arange(len(x))
            valid_idx = indices[~mask]
            valid_x = x[~mask]
            data[mask, col] = np.interp(indices[mask], valid_idx, valid_x)
    return data

def fill_time_weighted_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Fill missing values using time-weighted mean."""
    data = np.copy(data)
    weights = np.linspace(1, 0.1, window)[::-1]
    weights /= weights.sum()
    
    for col in range(data.shape[1]):
        x = data[:, col]
        mask = np.isnan(x)
        for i in np.where(mask)[0]:
            past_idx = np.where(~np.isnan(x[:i+1]))[0]
            future_idx = np.where(~np.isnan(x[i:]))[0] + i
            past = x[past_idx[-window:]] if len(past_idx) > 0 else np.array([])
            future = x[future_idx[:window]] if len(future_idx) > 0 else np.array([])
            values = np.concatenate([past, future])
            if len(values) > 0:
                w = weights[:len(values)]
                data[i, col] = np.average(values, weights=w)
    return data


def fill_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Fill missing values using moving average."""
    data = np.copy(data)
    for col in range(data.shape[1]):
        x = data[:, col]
        mask = np.isnan(x)
        if mask.any():
            df = pd.Series(x)
            ma = df.rolling(window=window, min_periods=1, center=True).mean().values
            data[mask, col] = ma[mask]
    return data



FILL_METHODS: dict[str, Callable] = {
    "none": lambda data: data,
    "ffill": ffill,
    "bfill": bfill,
    "fill_csm": fill_cross_sectional_mean,
    "linear_interp": fill_linear_interd,
    "twm": lambda data: fill_time_weighted_mean(data, window=5),
    "movingm": lambda data: fill_moving_average(data, window=5),
}


def apply_fillingdata(
    data: pd.DataFrame,
    method: str,
) -> np.ndarray:
    """ Fill missing values in a DataFrame grouped by a specific level of MultiIndex."""
    fill_func = FILL_METHODS.get(method.lower())
    if fill_func is None:
        raise ValueError(f"Fill method '{method}' is not supported. Available methods: {list(FILL_METHODS.keys())}")
    
    filled_array = fill_func(data)
    filled_data = pd.DataFrame(filled_array, index=data.index, columns=data.columns)
    return filled_data



def analyze_fill_correlation(
    group_data:list,
    fill_methods: List[str],
    corr_type: str = "pearson",
    axis: int = 0,
    save_dir: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Analyze the correlation of filled data with the original data.
    Args:
        group_data: 
            List of DataFrames or 2D numpy arrays grouped by a specific level of MultiIndex.
        fill_methods:       
            List of fill methods to apply. Supported methods are 'ffill', 'bfill', 'fill_cross_sectional_mean',
            'linear_interpolation', 'time_weighted_mean', and 'moving_average'. 
        corr_type:
            Type of correlation to compute. Options are 'pearson', 'spearman', or 'kendall'.
        axis:
            Axis along which to compute the correlation. 0 for columns, 1 for rows.
    
    Optional:
        save_dir: 
            Directory to save the correlation results. If None, results are not saved.
    Returns:
        Dictionary with fill methods as keys and their correlation statistics as values.
        Dictionary with fill methods as keys and filled data arrays as values.
    Raises:
        ValueError: If original_data is not a DataFrame or if its index is not unique.
        ValueError: If corr_type is not one of 'pearson', 'spearman', or 'kendall'.

    """

    corr_func = {
        "pearson": pearsonr,
        "spearman": spearmanr,
        "kendall": kendalltau
    }.get(corr_type.lower())
    if corr_func is None:
        raise ValueError(f" upper corr_type: {corr_type} is not supported, select pearson, spearman or kendall")

    results = {}
    filled_data_cache = {}
    for fill_method in fill_methods:
        try:
            filled_array = [apply_fillingdata(data, fill_method) for data in group_data]
            filled_data_cache[fill_method] = filled_array 
            # Compute correlation for each row, ignoring NaNs
            valid_corr = pd.DataFrame(np.vstackfilled_array).corrwith(
                pd.DataFrame(np.vstack(group_data)),axis=1, method=corr_type) 
            results[fill_method] = {
                "mean_corr": valid_corr.mean() if not valid_corr.empty else 0.0,
                "std_corr": valid_corr.std() if not valid_corr.empty else 0.0,
                "median_corr": valid_corr.median() if not valid_corr.empty else 0.0,
                "min_corr": valid_corr.min() if not valid_corr.empty else 0.0,
                "max_corr": valid_corr.max() if not valid_corr.empty else 0.0,
                "n_valid": len(valid_corr)
            }
        except Exception as e:
            print(f"filling_method {fill_method} calculation failed: {e}")
            results[fill_method] = {
                "mean_corr": np.nan,
                "std_corr": np.nan,
                "median_corr": np.nan,
                "min_corr": np.nan,
                "max_corr": np.nan,
                "n_valid": 0,
                "mean_diff": np.nan,
                "std_diff": np.nan
            }

    return results, filled_data_cache

def select_robust_fill_methods(
    corr_results: Dict[str, Dict[str, float]],
    corr_threshold: float = 0.7,
    std_threshold: float = 0.1,
    min_valid_ratio: float = 0.8,
    max_methods: int = 3) -> List[str]:
    """
    Select robust fill methods based on correlation statistics.
    Args:
        corr_results: 
            Dictionary with fill methods as keys and their correlation statistics as values.
        corr_threshold: 
            Minimum mean correlation to consider a method robust.
        std_threshold: 
            Maximum standard deviation of correlation to consider a method robust.
        min_valid_ratio: 
            Minimum ratio of valid correlations to total columns to consider a method robust.
        max_methods: 
            Maximum number of robust methods to return.
    Returns:
        List of robust fill methods sorted by mean correlation, standard deviation, and mean difference.
    """
    if not corr_results:
        return []

    # calculate total number of columns for valid ratio
    total_cols = max([stats["n_valid"] for stats in corr_results.values() if stats["n_valid"] > 0], default=1)

    # filter methods based on thresholds
    robust_methods = []
    for method, stats in corr_results.items():
        if (
            stats["mean_corr"] >= corr_threshold and
            stats["std_corr"] <= std_threshold and
            stats["n_valid"] / total_cols >= min_valid_ratio
        ):
            robust_methods.append((method, stats["mean_corr"], stats["std_corr"], stats["mean_diff"]))

    # sort methods by mean correlation, then by std deviation, then by mean difference
    robust_methods.sort(key=lambda x: (-x[1], x[2], x[3]))

    # keep only the method names up to max_methods
    return [method for method, _, _, _ in robust_methods[:max_methods]]

