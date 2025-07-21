"""
Module: alpha_checks
This module provides the `FilterAlphaChecks` class, which contains methods to validate various alpha metrics against configured thresholds. The checks include date validation, Information Coefficient (IC), cover ratio, returns, ICIR, Sharpe ratio, long returns, turnover rate (TVR), drawdown (DD), and win rate. Each check method compares the input metric to the corresponding threshold defined in a `FilterAlphaConfig` object.
Classes:
    FilterAlphaChecks: Provides methods to check alpha metrics against configuration thresholds.
Methods:
    __init__(self, config: FilterAlphaConfig)
        Initializes the FilterAlphaChecks instance with a configuration object.
    check_from(self, from_date: Union[str, int, None]) -> bool
        Checks if the provided from_date matches the configured start_date.
    check_ic(self, ic: float) -> int
        Checks the Information Coefficient (IC) against the configured threshold.
    check_cover_ratio(self, long_ratio: float) -> int
        Checks the long cover ratio against the configured threshold.
    check_ret(self, ret: float) -> bool
        Checks if the return meets or exceeds the configured threshold.
    check_icir(self, icir: float) -> bool
        Checks if the ICIR meets or exceeds the configured threshold.
    check_sharpe(self, sharpe: float) -> bool
        Checks if the Sharpe ratio meets or exceeds the configured threshold.
    check_ret_l(self, ret_l: float) -> bool
        Checks if the long return meets or exceeds the configured threshold.
    check_tvr(self, tvr: float) -> int
        Checks if the turnover rate (TVR) is within the configured min and max thresholds.
    check_dd(self, dd: float) -> bool
        Checks if the drawdown is less than or equal to the configured threshold.
    check_win(self, win: float) -> bool
        Checks if the win rate meets or exceeds the configured threshold.
    check 
"""

import logging
from typing import Union
import numpy as np
from datetime import datetime
from .config_validation import FilterAlphaConfig

# Configure logging
logger = logging.getLogger(__name__)

class FilterAlphaChecks:
    """Class for checking alpha metrics against configured thresholds."""
    def __init__(self, config: FilterAlphaConfig):
        """Initialize with configuration parameters.

        Args:
            config: FilterAlphaConfig object containing threshold parameters.
        """
        self.config = config

    def check_from(self, from_date: Union[str, int, None]) -> bool:
        """Check if the from_date is equal to the configured start_date.

        Args:
            from_date (any): The start date of the data, can be YYYYMMDD (int/str), YYYY-MM-DD (str), or YYYY/MM/DD (str).

        Returns:
            bool: True if from_date equals start_date, False otherwise.

        Raises:
            TypeError: If from_date is neither str nor int.
        """
        if from_date is None:
            logger.warning("None from_date provided, returning False")
            return False

        # Convert to string if input is int
        if isinstance(from_date, int):
            from_date = str(from_date)

        if not isinstance(from_date, str):
            logger.error(f"Invalid type for from_date: {type(from_date)}, expected str or int")
            raise TypeError(f"from_date must be a string or integer, got {type(from_date)}")

        # Convert date to YYYYMMDD integer
        try:
            # Try multiple date formats
            formats = ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']
            date_str = from_date
            for fmt in formats:
                try:
                    date_obj = datetime.strptime(from_date, fmt)
                    date_str = date_obj.strftime('%Y%m%d')
                    break
                except ValueError:
                    continue

            if not date_str or (date_str == from_date and not from_date.isdigit()):  # If no valid format found
                logger.warning(f"Could not parse from_date: {from_date} to YYYYMMDD, returning False")
                return False

            from_num = int(date_str)
        except ValueError:
            logger.warning(f"Could not convert from_date: {from_date} to integer, returning False")
            return False

        # Convert config start_date to YYYYMMDD integer
        try:
            start_date_obj = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            start_num = int(start_date_obj.strftime('%Y%m%d'))
        except ValueError as e:
            logger.error(f"Invalid start_date format in config: {self.config.start_date}")
            raise ValueError(f"Invalid start_date format in config: {self.config.start_date}: {str(e)}")

        return from_num == start_num

    def check_ic(self, ic: float) -> int:
        """Check Information Coefficient (IC).

        Args:
            ic (float): IC value.

        Returns:
            int: 1 if IC >= threshold, -1 if IC <= -threshold, 0 otherwise.
        """
        if not isinstance(ic, (float, int)) or np.isnan(ic):
            return 0
        if ic >= self.config.ic:
            return 1
        elif ic <= -self.config.ic:
            return -1
        else:
            return 0

    def check_cover_ratio(self, long_ratio: float) -> int:
        """Check long cover ratio.

        Args:
            long_ratio (float): Long cover ratio.

        if long_ratio >= self.config.cover:
            return 1
        elif long_ratio < self.config.cover:
            return -1
        else:
            return 0
            int: 1 if ratio >= threshold, -1 if ratio < threshold, 0 otherwise.
        """
        if np.isnan(long_ratio):
            return 0
        return 1 if long_ratio >= self.config.cover else -1 if long_ratio < self.config.cover else 0

    def check_ret(self, ret: float) -> bool:
        """Check return.

        Args:
            ret (float): Return value.

        Returns:
            bool: True if ret >= threshold, False otherwise.
        """
        return ret >= self.config.ret

    def check_icir(self, icir: float) -> bool:
        """Check ICIR.

        Args:
            icir (float): ICIR value.

        Returns:
            bool: True if ICIR >= threshold, False otherwise.
        """
        return icir >= self.config.icir

    def check_sharpe(self, sharpe: float) -> bool:
        """Check Sharpe ratio.

        Args:
            sharpe (float): Sharpe ratio.

        Returns:
            bool: True if Sharpe >= threshold, False otherwise.
        """
        return sharpe >= self.config.sharpe

    def check_ret_l(self, ret_l: float) -> bool:
        """Check long return.

        Args:
            ret_l (float): Long return value.

        Returns:
            bool: True if ret_l >= threshold, False otherwise.
        """
        return ret_l >= self.config.ret_l

    def check_tvr(self, tvr: float) -> int:
        """Check Turnover Rate (TVR).

        Args:
            tvr (float): TVR value.

        Returns:
        if tvr is None or not isinstance(tvr, (float, int)):
            logger.warning(f"Invalid tvr value: {tvr}, expected a float or int")
        if tvr > self.config.tvr_max:
            return -1
        elif tvr < self.config.tvr_min:
            return 1
        else:
            return 0
        if np.isnan(tvr):
        """
        if np.isnan(tvr):
            return 0
        return -1 if tvr > self.config.tvr_max else 1 if tvr < self.config.tvr_min else 0

    def check_dd(self, dd: float) -> bool:
        """Check Drawdown (DD).

        Args:
            dd (float): Drawdown value.
        Returns:
            bool: True if DD <= threshold, False otherwise.
        """
        if dd is None or not isinstance(dd, (float, int)):
            logger.warning(f"Invalid dd value: {dd}, expected a float or int. Returning False.")
            return False
        return float(dd) <= self.config.dd

    def check_win(self, win: float) -> bool:
        """
        Check win rate.

        Args:
            win (float): Win rate.

        Returns:
            bool: True if win >= threshold, False otherwise.
        """
        if win is None or not isinstance(win, (float, int)):
            logger.warning(f"Invalid win value: {win}, expected a float or int. Returning False.")
            return False
        return float(win) >= self.config.win
