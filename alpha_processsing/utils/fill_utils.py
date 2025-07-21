
from bz2 import compress
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.impute import KNNImputer


def ffill(data: np.ndarray) -> np.ndarray:
    data = np.copy(data)
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(data.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    row_indices = idx[mask]  # Shape: (num_nans,)
    col_indices = np.where(mask)[1]  # Shape: (num_nans,)
    data[mask] = data[row_indices, col_indices]
    return data

def bfill(data: np.ndarray) -> np.ndarray:
    data = np.copy(data)
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(data.shape[0])[:, None], data.shape[0] - 1)
    np.minimum.accumulate(idx[::-1], axis=0, out=idx[::-1])
    row_indices = idx[mask]          # Shape: (num_nans,)
    col_indices = np.where(mask)[1]  # Shape: (num_nans,)
    data[mask] = data[row_indices, col_indices]
    return data

def fill_cross_sectional_mean(data: np.ndarray) -> np.ndarray:
    """Fill missing values with the mean of each row (cross-sectional mean)."""
    data = np.copy(data)
    row_means = np.nanmean(data, axis=1, keepdims=True)
    mask = np.isnan(data)
    data[mask] = np.broadcast_to(row_means, data.shape)[mask]
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

def fill_time_weighted_mean(data: np.ndarray) -> np.ndarray:
    """Fill missing values using time-weighted mean."""
    col_means = np.nanmean(data, axis=0)
    nan_indices = np.isnan(data)
    filled_data = np.where(nan_indices, col_means, data)
    return data

def fill_knn_imputation(data: np.ndarray, k: int = 5) -> np.ndarray:
    """
    使用 K-近邻 (K-Nearest Neighbors) 算法填充缺失值。

    该方法通过寻找每个缺失点的 K 个最近邻样本，并用这些邻居的
    特征均值来填充缺失值。

    Args:
        data (np.ndarray): 输入的 NumPy 数组，包含 NaN 值。
        k (int): 用于填充的邻居数量。

    Returns:
        np.ndarray: 填充了缺失值的新数组。
    """
    if np.isnan(data).sum() == 0:
        return data.copy()
    is_all_nan_col = np.all(np.isnan(data), axis=0)
    all_nan_col_indices = np.where(is_all_nan_col)[0]
    good_col_indices = np.where(~is_all_nan_col)[0]
  
    if good_col_indices.size == 0:
        return data.copy()
    data_to_impute = data[:, good_col_indices]
    imputer = KNNImputer(n_neighbors=k, weights='uniform')
    filled_good_data = imputer.fit_transform(data_to_impute)
    final_data = np.full_like(data, fill_value=np.nan)
    final_data[:, good_col_indices] = filled_good_data  
    return final_data


        

FILL_METHODS: dict[str, Callable] = {
    "none": lambda data: data,
    "ffill": ffill,
    "bfill": bfill,
    "fill_cross_sectional_mean": fill_cross_sectional_mean,
    "fill_linear_interp": fill_linear_interd,
    "fill_time_weighted_mean": fill_time_weighted_mean,
    "fill_knn_imputation": lambda data: fill_knn_imputation(data, k=5)

}


def apply_fillingdata(
    data: pd.DataFrame,
    method: str,
) -> np.ndarray:
    """ Fill missing values in a DataFrame grouped by a specific level of MultiIndex."""
    fill_func = FILL_METHODS.get(method.lower())
    if fill_func is None:
        raise ValueError(f"Fill method '{method}' is not supported. Available methods: {list(FILL_METHODS.keys())}")
    
    filled_array = fill_func(data.copy())
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
    origin_df = pd.DataFrame(np.vstack(group_data))
    results = {}
    filled_data_cache = {}
    for fill_method in fill_methods:
        try:
            filled_array = [apply_fillingdata(data, fill_method) for data in group_data]
            filled_data_cache[fill_method] = filled_array.copy()
            # Compute correlation for each row, ignoring NaNs
            valid_corr = pd.DataFrame(np.vstack(filled_array)).corrwith(origin_df,axis=1, method=corr_type) 
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
    corr_threshold: float = 0.95,
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
            stats["mean_corr"] <= corr_threshold and
            stats["std_corr"] <= std_threshold and
            stats["n_valid"] / total_cols >= min_valid_ratio
        ):
            robust_methods.append((method, stats["mean_corr"], stats["std_corr"], stats["mean_diff"]))
    if len(robust_methods) == 0:
        return ["none"]
    # sort methods by mean correlation, then by std deviation, then by mean difference
    robust_methods.sort(key=lambda x: (-x[1], x[2], x[3]))

    # keep only the method names up to max_methods
    return [method for method, _, _, _ in robust_methods[:max_methods]]



