"""
This module provides functions for reading and processing statistical CSV files related to alpha factor backtesting.
Functions:
    _read_stats_csv(path: str, index_col: int = 0, exclude_last: bool = False) -> pd.DataFrame
        Reads a stats CSV file and returns a DataFrame with selected columns. Handles different index columns and can exclude the last row if needed.
    read_tvr_ic(path: str) -> pd.Series
        Reads TVR/IC statistics from a CSV file and returns the stats for the 'Total' row as a Series.
    read_tvr_year(path: str) -> pd.DataFrame
        Reads yearly TVR/IC statistics from a CSV file, excluding the last row, and returns them as a DataFrame.
    read_multiple_tvr_ic(paths: List[str]) -> pd.DataFrame
        Reads TVR/IC statistics from multiple CSV files, concatenates them into a single DataFrame, and adds an 'alpha' column indicating the alpha name.
    FileOperationError: Raised when a file cannot be read, is not found, or has an invalid format.
"""
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import List
from .file_operations import FileOperationError

# Configure logging
logger = logging.getLogger(__name__)

# Configure path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir.resolve()))

from path_set import backtest_path

def _read_stats_csv(path: str, index_col: int = 0, exclude_last: bool = False) -> pd.DataFrame:
    """Base function to read stats CSV.

    Args:
        path (str): Path to the CSV file.
        index_col (int): Index column number.
        exclude_last (bool): Whether to exclude the last row.

    Returns:
        pd.DataFrame: DataFrame with selected columns.

    Expected CSV Format:
        The CSV file should contain the following columns:
        - 'from': Start date of the period.
        - 'to': End date of the period.
        - 'ret': Return.
        - 'ret_l': Return for long positions.
        - 'ret_s': Return for short positions.
        - 'IC': Information Coefficient.
        - 'ICIR': Information Coefficient Information Ratio.
        - 'tvr': Turnover rate.
        - 'sharpe': Sharpe ratio.
        - 'dd%': Drawdown percentage.
        - 'pwin': Probability of winning.
        - 'long': Long positions.
        - 'short': Short positions.

    Raises:
        FileOperationError: If the file cannot be read or is invalid.
    """
    try:
        df = pd.read_csv(path, index_col=index_col)
        if exclude_last:
            df = df.iloc[:-1]
        required_columns = ['from', 'to', 'ret', 'ret_l', 'ret_s', 'IC', 'ICIR',
                            'tvr', 'sharpe', 'dd%', 'pwin', 'long', 'short']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            if index_col == 0:
                return _read_stats_csv(path, index_col=1, exclude_last=exclude_last)
            raise FileOperationError(f"Invalid index column in {path}")
        return df[[col for col in required_columns if col in df.columns]]
    except KeyError:
        if index_col == 0:
            return _read_stats_csv(path, index_col=1, exclude_last=exclude_last)
        raise FileOperationError(f"Invalid index column in {path}")
    except FileNotFoundError:
        raise FileOperationError(f"Stats file not found: {path}")
    except Exception as e:
        raise FileOperationError(f"Error reading file {path}: {str(e)}")

def read_tvr_ic(path: str) -> pd.Series:
    """Read TVR/IC stats for 'Total' row.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.Series: Series containing stats for the 'Total' row.

    Raises:
        FileOperationError: If the file cannot be read.
    """
    return _read_stats_csv(path, index_col=0).loc['Total']

def read_tvr_year(path: str) -> pd.DataFrame:
    """Read yearly TVR/IC stats, excluding the last row.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing yearly stats with the following columns:
            - 'from': Start date of the period.
            - 'to': End date of the period.
            - 'ret': Return.
            - 'ret_l': Return for long positions.
            - 'ret_s': Return for short positions.
            - 'IC': Information Coefficient.
            - 'ICIR': Information Coefficient Information Ratio.
            - 'tvr': Turnover rate.
            - 'sharpe': Sharpe ratio.
            - 'dd%': Drawdown percentage.
            - 'pwin': Probability of winning.
            - 'long': Long positions.
            - 'short': Short positions.

    Raises:
        FileOperationError: If the file cannot be read.
    """
    return _read_stats_csv(path, index_col=0, exclude_last=True)

def read_multiple_tvr_ic(paths: List[str]) -> pd.DataFrame:
    """Read multiple TVR/IC stats into a single DataFrame.

    Args:
        paths (List[str]): List of paths to CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with alpha names.

    Raises:
        FileOperationError: If any file cannot be read.
    """
    results = []
    for path in paths:
        try:
            stats = read_tvr_ic(path)
            stats['alpha'] = Path(path).parent.parent.parent.stem
            results.append(stats)
        except FileOperationError as e:
            logger.warning(str(e))
            continue
    if not results:
        raise FileOperationError("No valid stats files found")
    return pd.concat(results, axis=1).T