
from bz2 import compress
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from typing import Union, Optional, Callable
from pathlib import Path
import matplotlib.pyplot as plt


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

def fill_csm(data: np.ndarray) -> np.ndarray:
    """Fill missing values with the mean of each row (cross-sectional mean)."""
    data = np.copy(data)
    row_means = np.nanmean(data, axis=1, keepdims=True)
    mask = np.isnan(data)
    data[mask] = row_means[mask]
    return data

def fill_null(data: np.ndarray) -> np.ndarray:
    return np.copy(data)

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

def fill_twm(data: np.ndarray, window: int) -> np.ndarray:
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


def fill_knn(data: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Fill missing values using KNN imputation."""
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    return imputer.fit_transform(data)

def fill_movingm(data: np.ndarray, window: int) -> np.ndarray:
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

def fill_pca(data: np.ndarray, n_components: int = 5) -> np.ndarray:
    """Fill missing values using a factor model with PCA."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std[std == 0] = 1
    standardized = (data - mean) / std
    standardized[np.isnan(standardized)] = 0
    pca = PCA(n_components=min(n_components, data.shape[1]))
    factors = pca.fit_transform(standardized)
    loadings = pca.components_.T
    reconstructed = np.dot(factors, loadings.T)
    reconstructed = reconstructed * std + mean
    mask = np.isnan(data)
    data = np.copy(data)
    data[mask] = reconstructed[mask]
    return data

FILL_METHODS: dict[str, Callable] = {
    "ffill": ffill,
    "bfill": bfill,
    "fill_cross_sectional_mean": fill_csm,
    "null": fill_null,
    "linear_interpolation": fill_linear_interd,
    "time_weighted_mean": lambda data: fill_twm(data, window=5),
    "knn_imputation": fill_knn,
    "moving_average": lambda data: fill_movingm(data, window=5),
    "factor_model": fill_pca
}

def fill_data_by_group(
    data: pd.DataFrame,
    method: str,
    window: Optional[int] = None,
    group_level: str = "group_id"
) -> np.ndarray:
    """ Fill missing values in a DataFrame grouped by a specific level of MultiIndex."""
    if not isinstance(data.index, pd.MultiIndex) or group_level not in data.index.names:
        raise ValueError(f"data has to be a DataFrame with MultiIndex and group_level must be one of the index names: {data.index.names}")

    if method not in FILL_METHODS:
        raise ValueError(f"unsupported method: {method}")

    result = data.copy()
    try:
        fill_func = FILL_METHODS[method]
        for group_id in data.index.get_level_values("group_id").unique():
            group_data = data.xs(group_id, level="group_id")
            if method in ["time_weighted_mean", "moving_average"] and window is not None:
                filled = fill_func(group_data.values, window=window)
            else:
                filled = fill_func(group_data.values)
            result.loc[(slice(None), group_id), :] = filled
    except Exception as e:
        print(f"filling method {method} failed: {e}")
    return result.values



def analyze_fill_correlation(
    original_data: pd.DataFrame,
    fill_methods: List[str],
    corr_type: str = "pearson",
    axis: int = 0,
    save_dir: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Analyze the correlation of filled data with the original data.
    Args:
        original_data: 
            original DataFrame with unique index.
        fill_methods:
            List of fill methods to apply
        corr_type:
            Type of correlation to compute ('pearson', 'spearman', 'kendall').
        axis:
            Axis along which to compute correlation (0 for columns, 1 for rows).
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
    if not isinstance(original_data, pd.DataFrame):
        raise ValueError("oringal_data  has to be a pandas DataFrame")
    if not original_data.index.is_unique:
        raise ValueError("original_data with index must have unique index values")

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
            filled_array = fill_data_by_group(original_data, fill_method, window=5)
            filled_data = pd.DataFrame(filled_array, index=original_data.index, columns=original_data.columns)
            filled_data_cache[fill_method] = filled_data
            corr_matrix = filled_data.corrwith(original_data, axis=axis, method=corr_type)
            valid_corr = corr_matrix[~corr_matrix.isna()]

            results[fill_method] = {
                "mean_corr": valid_corr.mean() if not valid_corr.empty else 0.0,
                "std_corr": valid_corr.std() if not valid_corr.empty else 0.0,
                "median_corr": valid_corr.median() if not valid_corr.empty else 0.0,
                "min_corr": valid_corr.min() if not valid_corr.empty else 0.0,
                "max_corr": valid_corr.max() if not valid_corr.empty else 0.0,
                "n_valid": len(valid_corr),
                "mean_diff": np.nanmean(np.abs(filled_array - original_data.values)),
                "std_diff": np.nanstd(filled_array - original_data.values)
            }
            
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                filepath = save_path / f"filled_{fill_method}.parquet"
                filled_data.to_parquet(filepath, compression='zstd', index=True, engine='pyarrow')
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
    corr_threshold: float = 0.9,
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

