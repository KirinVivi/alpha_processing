
"""
filtering.py
This module provides the FilterAlpha class and related utilities for filtering, categorizing, and analyzing alpha strategies based on backtest results. It includes methods for reading and caching alpha statistics, categorizing alphas according to configurable rules, analyzing overall and category-specific metrics, identifying extremes, saving results in various formats, plotting category distributions, fixing alpha files, and transferring passing alphas.
Classes:
    CategoryRule: Defines a rule for categorizing alphas based on a condition.
    FilterAlpha: Main class for filtering and analyzing alpha strategies.
FilterAlpha Methods:
    __init__(self, config: FilterAlphaConfig)
        Initializes the FilterAlpha instance with the provided configuration.
    _get_cached_stats(self, path: str) -> pd.DataFrame
        Retrieves and caches yearly statistics DataFrame for a given path.
    _get_cached_ic(self, path: str) -> pd.Series
        Retrieves and caches IC summary Series for a given path.
    run_backtest_file(self, backtest_folder: str) -> Tuple[List[str], List[str]]
        Processes a backtest folder to extract alpha names and statistics file paths.
    _get_alpha_stats(self, folder_path: Path) -> Tuple[List[str], List[str]]
        Extracts alpha names and statistics paths from a nested folder structure.
    _categorize_alpha(self, alpha_name: str, result: pd.Series, path: Path) -> Tuple[Optional[str], Optional[str]]
        Categorizes an alpha based on its statistics and predefined rules.
    analyze_overall_metrics(self, stats_paths: List[str]) -> pd.DataFrame
        Analyzes overall yearly metrics across all alphas.
    analyze_category_metrics(self, stats_paths: List[str], category_dict: Dict[str, List[str]]) -> pd.DataFrame
        Analyzes yearly metrics for each alpha category.
    analyze_category_extremes(self, stats_paths: List[str], category_dict: Dict[str, List[str]]) -> pd.DataFrame
        Analyzes maximum and minimum metrics for all alphas and categories, including yearly and total extremes.
    run_filter_backtest_average(self, alpha_list: List[str], stats_path: List[str]) -> None
        Filters the alpha list based on backtest averages and categorizes them.
    save_category_results(self, output_path: Path) -> None
        Saves metrics and extremes to CSV, and generates HTML/Markdown reports.
    plot_category_distribution(self) -> None
        Plots and saves the distribution of alpha categories.
    fix_alpha_files(self) -> None
        Fixes alpha_cls.py files for rerun alphas by adjusting TVR parameters.
    trans_test_compare(self) -> None
        Transfers or updates passing alphas to the target directory and updates records.
Exceptions:
    FileOperationError: Raised for file operation failures.
Dependencies:
    - pandas
    - matplotlib
    - tqdm
    - jinja2
    - logging
    - pathlib
    - re
    - time
    - os, sys
"""
import os
import sys
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from .config_validation import FilterAlphaConfig, validate_config
from .file_operations import (
    rmtree_alpha_file, transfer_alpha, update_total_alpha,
    backup_alpha, fix_nested_path, compare_alpha_cls, move_alpha_directory, FileOperationError
)
from .alpha_checks import FilterAlphaChecks
from .data_reading import read_tvr_ic, read_tvr_year

# Configure logging
logger = logging.getLogger(__name__)

# Configure path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir.resolve()))

from path_set import (
    backtest_path, update_alpha_path, update_pnl_path, update_summary_path,
    pnl_record_path, summary_record_path, alpha_hdf_path, output_pnl_path, output_af_path
)

class CategoryRule:
    """Rule for categorizing alphas based on conditions."""
    def __init__(self, condition: Callable[[Union[Dict, pd.Series]], bool], category: Optional[str]):
        self.condition = condition
        self.category = category

class FilterAlpha:
    def __init__(self, config: FilterAlphaConfig):
        """Initialize FilterAlpha with configuration."""
        self.config = validate_config(config) # type: ignore
        self.checks = FilterAlphaChecks(self.config)
        self.pass_dict: Dict[str, List[str]] = {}
        self.transfer_lis: List[str] = []
        self.keep_dict: Dict[str, List[str]] = {}
        self.rerun_lis: List[str] = []
        self.rerun_dict: Dict[str, List[str]] = {}
        self.run_lis: List[str] = []
        self.drop_lis: List[str] = []
        self.alpha_to_path: Dict[str, str] = {}
        # Caches for read_tvr_year and read_tvr_ic results
        self._stats_cache: Dict[str, pd.DataFrame] = {}  # Cache for read_tvr_year results
        self._ic_cache: Dict[str, pd.Series] = {}  # Cache for read_tvr_ic results

    def _get_cached_stats(self, path: str) -> pd.DataFrame:
        """Get cached stats DataFrame or read from file."""
        # Check if path is already cached
        if path not in self._stats_cache:
            try:
                df = read_tvr_year(path)
                self._stats_cache[path] = df
                logger.debug(f"Read and cached stats for {path}")
            except Exception as e:
                logger.error(f"Failed to read {path}: {e}")
                self._stats_cache[path] = pd.DataFrame()  # Cache empty DF on error
        return self._stats_cache[path]

    def _get_cached_ic(self, path: str) -> pd.Series:
        """Get cached IC Series or read from file."""
        # Check if path is already cached
        if path not in self._ic_cache:
            try:
                series = read_tvr_ic(path)
                self._ic_cache[path] = series
                logger.debug(f"Read and cached IC for {path}")
            except Exception as e:
                logger.error(f"Failed to read IC {path}: {e}")
                self._ic_cache[path] = pd.Series()  # Cache empty Series on error
        return self._ic_cache[path]

    def run_backtest_file(self, backtest_folder: str) -> Tuple[List[str], List[str]]:
        """Process backtest folder to extract alpha names and stats paths."""
        folder_path = Path(backtest_path) / backtest_folder
        if not folder_path.exists():
            raise FileOperationError(f"Backtest folder {folder_path} does not exist")
        return self._get_alpha_stats(folder_path)

    def _get_alpha_stats(self, folder_path: Path) -> Tuple[List[str], List[str]]:
        """Extract alpha names and stats paths from folder with nested out_decay/MyAlpha structure."""
        if not folder_path.exists():
            raise FileOperationError(f"Folder {folder_path} does not exist")
        start_time = time.time()
        alpha_name_cond = lambda x: not x.startswith('__') and not x.endswith('.py')
        alphas = [x for x in os.listdir(folder_path) if alpha_name_cond(x) and (folder_path / x).is_dir()]
        stats_paths = [str(p) for p in folder_path.rglob('out_decay/MyAlpha/yearly_stats_info_sum_alpha_ac.csv')]
        logger.info(f"Scanned {len(stats_paths)} stats files in {folder_path} in {time.time() - start_time:.3f}s")
        return alphas, stats_paths

    def _categorize_alpha(self, alpha_name: str, result: pd.Series, path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Categorize alpha based on check conditions."""
        required_fields = ['from', 'long', 'IC', 'tvr', 'ret', 'dd%', 'ICIR', 'pwin', 'ret_l']
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            logger.error(f"Missing fields {missing_fields} in stats for alpha {alpha_name}")
            return 'drop', None
        # set category rules
        rules = [
            CategoryRule(lambda r: not self.checks.check_from(r['from']), None),
            CategoryRule(lambda r: self.checks.check_cover_ratio(r['long']) == 0, None),
            CategoryRule(lambda r: self.checks.check_cover_ratio(r['long']) == -1, 'drop'),
            CategoryRule(lambda r: self.checks.check_ic(r['IC']) == 0 and (not self.checks.check_tvr(r['tvr']) == 1), 'drop'),
            CategoryRule(lambda r: self.checks.check_ic(r['IC']) == -1, 'negative_ic'),
            CategoryRule(lambda r: self.checks.check_tvr(r['tvr']) == 1, 'low_tvr'),
            CategoryRule(lambda r: self.checks.check_tvr(r['tvr']) == -1, 'high_tvr'),
            CategoryRule(lambda r: self.checks.check_ret(r['ret']) and self.checks.check_dd(r['dd%']) and
                        self.checks.check_icir(r['ICIR']) and self.checks.check_win(r['pwin']) and
                        self.checks.check_ret_l(r['ret_l']), 'all_pass'),
            CategoryRule(lambda r: self.checks.check_ret(r['ret']) and self.checks.check_dd(r['dd%']) and
                        self.checks.check_icir(r['ICIR']) and self.checks.check_win(r['pwin']), 'ret_long'),
            CategoryRule(lambda r: self.checks.check_ret(r['ret']) and self.checks.check_win(r['pwin']), 'high_dd'),
                CategoryRule(lambda r: self.checks.check_ic(r['IC']) and self.checks.check_ret_l(r['ret_l']), 'low_ret_s'), 
            CategoryRule(lambda r: self.checks.check_ret(r['ret']) and self.checks.check_dd(r['dd%']), 'low_win'),
            CategoryRule(lambda r: self.checks.check_ret(r['ret']), 'adjustable_ret'),
            CategoryRule(lambda r: self.checks.check_ret(r['ret'] + r['tvr'] / 35), 'adjustable_ret'),
        ]

        for rule in rules:
            try:
                if rule.condition(result):
                    return rule.category, str(path) if rule.category is None else None
            except ValueError as e:
                logger.warning(f"Failed to categorize alpha {alpha_name}: {e}")
                return None, str(path)
        return 'drop', None

    def analyze_overall_metrics(self, stats_paths: List[str]) -> pd.DataFrame:
        """Analyze overall yearly metrics across all alphas."""
        yearly_data, all_data = [], []
        metrics = ['ret', 'ret_l', 'IC', 'ICIR', 'tvr', 'sharpe', 'dd%', 'pwin']
        # Iterate through stats paths and collect yearly data
        for path in tqdm(stats_paths, desc="Analyzing overall metrics"):
            alpha_name = Path(path).parent.parent.parent.stem
            # yearly data from read_tvr_year
            df = self._get_cached_stats(path)
            if df.empty:
                logger.warning(f"No yearly data for {alpha_name}")
                continue
            df['year'] = pd.to_datetime(df['from'], format = '%Y%m%d', errors='coerce').dt.year
            if df['year'].isna().any():
                logger.error(f"Invalid data for {alpha_name}")
                continue
            yearly_data.append(df[['year'] + metrics])

            # Summary data from read_tvr_ic
            result = self._get_cached_ic(path)
            if result.empty:
                logger.warning(f"Invalid dates at {path}")
                continue
            data = {'year':f"{result.get('from')}-{result.get('to')}"}
            data.update({metric:result.get(metric, pd.NA) for metric in metrics})
            all_data.append(data)
        if not yearly_data and not all_data:
            logger.warning("No data for overall metrics analysis")
            return pd.DataFrame()
        # Calculate overall stats
        stats = []
        all_data_df = pd.DataFrame(all_data)
        overall_stats = {'year': 'total', 'count': len(all_data_df)}
        for metric in metrics:
                if metric in all_data:
                    overall_stats.update({
                        f"{metric}_mean": all_data_df[metric].mean(),
                        f"{metric}_std": all_data_df[metric].std() or 0,
                        f"{metric}_min": all_data_df[metric].min(),
                        f"{metric}_max": all_data_df[metric].max(),
                    })
        stats.append(overall_stats)
        # Concatenate yearly data and calculate yearly stats
        all_yearly_data = pd.concat(yearly_data, ignore_index=True)
        for year in all_yearly_data['year'].unique():
            year_data = all_yearly_data[all_yearly_data['year'] == year]
            year_stats = {'year': year,'count': len(year_data)}
            for metric in metrics:
                if metric in year_data:
                    year_stats.update({
                        f"{metric}_mean": year_data[metric].mean(),
                        f"{metric}_std": year_data[metric].std() or 0,
                        f"{metric}_min": year_data[metric].min(),
                        f"{metric}_max": year_data[metric].max(),
                    })
            stats.append(year_stats)
        return pd.DataFrame(stats)

    def analyze_category_metrics(self, stats_paths: List[str], category_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyze yearly metrics for each category."""
        alpha_to_category = {alpha: cat for cat, alphas in category_dict.items() for alpha in alphas}
        for path in self.run_lis:
            alpha_to_category[Path(path).parent.parent.parent.stem] = 'not_run'
        # Initialize lists for yearly and summary data
        yearly_data , summary_data = [], []
        # Define metrics to analyze
        metrics = ['ret', 'ret_l', 'ret_s',  'IC', 'ICIR', 'tvr', 'sharpe', 'dd%', 'pwin']
        
        # Iterate through stats paths and categorize alphas
        for path in tqdm(stats_paths, desc="Analyzing category metrics"):
            try:
                alpha_name = Path(path).parent.parent.parent.stem
                category = alpha_to_category.get(alpha_name, 'drop')
                if category == 'not_run':
                    continue
                # Read yearly data
                df = self._get_cached_stats(path)
                if df.empty:
                    logger.warning(f"No data for alpha {alpha_name}")
                    continue
                df['year'] = pd.to_datetime(df['from'], format='%Y/%m/%d', errors='coerce').dt.year
                if df['year'].isna().any():
                    logger.warning(f"Invalid dates for alpha {alpha_name}")
                    continue
                df['category'] = category
                yearly_data.append(df[['year', 'category'] + metrics])
                # Read summary data
                result = self._get_cached_ic(path)
                if result.empty:
                    logger.warning(f"No summary data for alpha {alpha_name}")
                    continue
                summary = {'alpha_name': alpha_name, 'category': category}
                summary.update({metric: result.get(metric, pd.NA) for metric in metrics})
                summary_data.append(summary)

            except Exception as e:
                logger.error(f"Error analyzing {path}: {e}")
                continue
        
        if not yearly_data and not summary_data:
            logger.warning("No data for category analysis")
            return pd.DataFrame()
        
        # Calculate category stats
        all_data = pd.concat(yearly_data, ignore_index=True)
        stats = []
        for (category, year), group in all_data.groupby(['category', 'year']):
            year_stats = {'category': category, 'year': year, 'count': len(group)}
            for metric in metrics:
                if metric in group:
                    year_stats.update({
                        f"{metric}_mean": group[metric].mean(),
                        f"{metric}_std": group[metric].std() or 0,
                        f"{metric}_min": group[metric].min(),
                        f"{metric}_max": group[metric].max()
                    })
            stats.append(year_stats)
        # Calculate overall category stats
        summary_stats = []
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            for category in summary_df['category'].unique():
                cat_data = summary_df[summary_df['category'] == category]
                cat_stats = {'category': category, 'count': len(cat_data), 'year': 'Total'}
                for metric in metrics:
                    if metric in cat_data:
                        cat_stats.update({
                            f"{metric}_mean": cat_data[metric].mean(),
                            f"{metric}_std": cat_data[metric].std() or 0,
                            f"{metric}_min": cat_data[metric].min(),
                            f"{metric}_max": cat_data[metric].max()
                        })
                summary_stats.append(cat_stats)
        # Combine overall and category stats
        if summary_stats:
            stats.extend(summary_stats)
        return pd.DataFrame(stats)

    def analyze_category_extremes(self, stats_paths: List[str], category_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyze max/min metrics for all alphas and categories with explicit Total index."""
        # Map alphas to categories
        alpha_to_category = {alpha: cat for cat, alphas in category_dict.items() for alpha in alphas}
        for path in self.run_lis:
            alpha_to_category[Path(path).parent.parent.parent.stem] = 'not_run'
        all_data, all_data_total = [], []
        # Define metrics to analyze
        metrics = ['ret', 'ret_l', 'ret_s',  'IC', 'ICIR', 'tvr', 'sharpe', 'dd%', 'pwin']
        # Collect data for extremes analysis
        for path in tqdm(stats_paths, desc="Collecting extremes data"):
            alpha_name = Path(path).parent.parent.parent.stem
            category = alpha_to_category.get(alpha_name, 'drop')
            if category == 'not_run':
                continue
            try:
                # Read yearly data
                df = self._get_cached_stats(path)
                if df.empty:
                    logger.warning(f"No data for alpha {alpha_name}")
                    continue
                df['year'] = pd.to_datetime(df['from'],format='%Y%m%d', errors='coerce').dt.year
                if df['year'].isna().any():
                    logger.warning(f"Invalid dates for alpha {alpha_name}")
                    continue
                df['alpha_name'] = alpha_name
                df['category'] = category
                all_data.append(df[['alpha_name', 'category', 'year'] + metrics])
                # Read summary data
                result = self._get_cached_ic(path)
                if result.empty:
                    logger.warning(f"No summary data for alpha {alpha_name}")
                    continue
                data = {'alpha_name': alpha_name, 'category': category, 'year': 'Total'}
                data.update({metric: result.get(metric, pd.NA) for metric in metrics})
                all_data_total.append(pd.DataFrame([data]))
            except Exception as e:
                logger.error(f"Error reading {alpha_name}: {e}")
                continue
    
        if not all_data and not all_data_total:
            logger.warning("No data for extremes analysis")
            return pd.DataFrame()

        all_data_df = pd.concat(all_data, ignore_index=True)
        all_data_total_df = pd.concat(all_data_total, axis=1).T.reset_index()
        extremes = []

        # 1. All alphas total extremes
        for metric in metrics:
            if metric in all_data_total_df:
                total_data = all_data_total_df[['alpha_name', metric]]
                max_val = total_data[metric].max()
                min_val = total_data[metric].min()
                max_alpha = total_data.loc[total_data[metric].idxmax()]['alpha_name'] if not total_data.loc[total_data[metric].idxmax()].empty else 'N/A'
                min_alpha = total_data.loc[total_data[metric].idxmin()]['alpha_name'] if not total_data.loc[total_data[metric].idxmin()].empty else 'N/A'
                extremes.append({
                    'scope': 'all_total',
                    'category': 'all',
                    'year': 'Total',  # Explicit Total index for aggregate metrics
                    'metric': metric,
                    'max_value': max_val,
                    'max_alpha': max_alpha,
                    'min_value': min_val,
                    'min_alpha': min_alpha
                })

        # 2. Category total extremes
        for category in category_dict.keys():
            cat_data = all_data_total_df[all_data_total_df['category'] == category]
            if cat_data.empty:
                continue
            for metric in metrics:
                if metric in cat_data:
                    total_data = cat_data[['alpha_name',metric]]
                    max_val = total_data[metric].max()
                    min_val = total_data[metric].min()
                    max_alpha = total_data.loc[total_data[metric].idxmax()]['alpha_name'] if not total_data.loc[total_data[metric].idxmax()].empty else 'N/A'
                    min_alpha = total_data.loc[total_data[metric].idxmin()]['alpha_name'] if not total_data.loc[total_data[metric].idxmin()].empty else 'N/A'
                    extremes.append({
                        'scope': 'category_total',
                        'category': category,
                        'year': 'Total',  # Explicit Total index for category aggregates
                        'metric': metric,
                        'max_value': max_val,
                        'max_alpha': max_alpha,
                        'min_value': min_val,
                        'min_alpha': min_alpha
                    })

        # 3. Yearly extremes for all alphas
        for year in all_data_df['year'].unique():
            year_data = all_data_df[all_data_df['year'] == year]
            for metric in metrics:
                if metric in year_data:
                    max_val = year_data[metric].max()
                    min_val = year_data[metric].min()
                    max_alpha = year_data[year_data[metric] == max_val]['alpha_name'].iloc[0] if not year_data[year_data[metric] == max_val].empty else 'N/A'
                    min_alpha = year_data[year_data[metric] == min_val]['alpha_name'].iloc[0] if not year_data[year_data[metric] == min_val].empty else 'N/A'
                    extremes.append({
                        'scope': 'all_yearly',
                        'category': 'all',
                        'year': year,
                        'metric': metric,
                        'max_value': max_val,
                        'max_alpha': max_alpha,
                        'min_value': min_val,
                        'min_alpha': min_alpha
                    })

        # 4. Yearly extremes for each category
        for category in category_dict.keys():
            cat_data = all_data_df[all_data_df['category'] == category]
            if cat_data.empty:
                continue
            for year in cat_data['year'].unique():
                year_cat_data = cat_data[cat_data['year'] == year]
                for metric in metrics:
                    if metric in year_cat_data:
                        max_val = year_cat_data[metric].max()
                        min_val = year_cat_data[metric].min()
                        max_alpha = year_cat_data[year_cat_data[metric] == max_val]['alpha_name'].iloc[0] if not year_cat_data[year_cat_data[metric] == max_val].empty else 'N/A'
                        min_alpha = year_cat_data[year_cat_data[metric] == min_val]['alpha_name'].iloc[0] if not year_cat_data[year_cat_data[metric] == min_val].empty else 'N/A'
                        extremes.append({
                            'scope': 'category_yearly',
                            'category': category,
                            'year': year,
                            'metric': metric,
                            'max_value': max_val,
                            'max_alpha': max_alpha,
                            'min_value': min_val,
                            'min_alpha': min_alpha
                        })

        extremes_df = pd.DataFrame(extremes)
        
        # Set MultiIndex for structured access
        extremes_df = extremes_df.set_index(['scope', 'year', 'category'])
        logger.info(f"Extremes DF indexed: {extremes_df.head().to_dict()}")
        
        # Reset index for CSV and report compatibility
        extremes_df = extremes_df.reset_index()
        return extremes_df

    def run_filter_backtest_average(self, alpha_list: List[str], stats_path: List[str]) -> None:
        """Filter alpha list based on backtest average."""
        if not alpha_list or not stats_path:
            raise ValueError("alpha_list and stats_path cannot be empty")
        # Initialize dictionaries to hold categorized alphas
        suff_pass = {key: [] for key in ['ret_long', 'sharpe', 'all_pass']}
        suff_keep = {key: [] for key in ['negative_ic', 'high_tvr', 'low_tvr', 'low_win', 'high_dd', 'adjustable_ret']}
        suff_drop = set()
        alpha_to_path = {Path(p).parent.parent.parent.stem: p for p in stats_path}
        suff_not_run = set(alpha_list) - set(alpha_to_path.keys())
        # Iterate through alphas and categorize them based on checks
        for alpha_name, path in tqdm(alpha_to_path.items(), desc="Filtering alphas"):
            result = read_tvr_ic(path)
            category, not_run_path = self._categorize_alpha(alpha_name, result, Path(path))

            if not_run_path:
                suff_not_run.add(not_run_path)
            elif category == 'drop':
                suff_drop.add(alpha_name)
            elif category in suff_pass:
                suff_pass[category].append(alpha_name)
            elif category in suff_keep:
                suff_keep[category].append(alpha_name)

        pass_alpha = list(set().union(*suff_pass.values()))
        keep_alpha = list(set().union(*suff_keep.values()))
        logger.info(f"Pass: {len(pass_alpha)}, Rerun: {len(keep_alpha)}, Drop: {len(suff_drop)}, Not run: {len(suff_not_run)}")

        self.pass_dict = suff_pass
        self.transfer_lis = pass_alpha
        self.keep_dict = suff_keep
        self.rerun_lis = sorted(keep_alpha)
        self.rerun_dict = suff_keep
        self.run_lis = list(suff_not_run)
        self.drop_lis = list(suff_drop)
        self.alpha_to_path = alpha_to_path
        # Save results to CSV
        self.save_category_results(Path(self.config.out_path) / 'category_results.csv')
        # clear caches
        self._stats_cache.clear()
        self._ic_cache.clear()
        logger.info('Cleared stats and IC caches after filtering')

    def save_category_results(self, output_path: Path) -> None:
        """Save metrics and extremes to CSV and generate HTML/Markdown reports."""
        try:
            # Analyze metrics
            category_dict = {
                'pass': self.transfer_lis,
                'rerun': self.rerun_lis,
                'drop': self.drop_lis,
                'not_run': [Path(path).parent.parent.parent.stem for path in self.run_lis]
            }
            overall_metrics_df = pd.DataFrame()
            category_metrics_df = pd.DataFrame()
            extremes_df = pd.DataFrame()
            if self.alpha_to_path:
                overall_metrics_df = self.analyze_overall_metrics(list(self.alpha_to_path.values()))
                category_metrics_df = self.analyze_category_metrics(list(self.alpha_to_path.values()), category_dict)
                extremes_df = self.analyze_category_extremes(list(self.alpha_to_path.values()), category_dict)

            # Save combined CSV
            combined_df = pd.DataFrame()
            if not overall_metrics_df.empty:
                combined_df = overall_metrics_df.copy()
                for metric in ['ret', 'IC', 'tvr', 'sharpe']:
                    for stat in ['mean', 'std', 'min', 'max']:
                        combined_df[f"overall_{metric}_{stat}"] = overall_metrics_df[f"{metric}_{stat}"]
            if not category_metrics_df.empty:
                for category in category_metrics_df['category'].unique():
                    cat_data = category_metrics_df[category_metrics_df['category'] == category]
                    for metric in ['ret', 'IC', 'tvr', 'sharpe']:
                        for stat in ['mean', 'std', 'min', 'max']:
                            for year in cat_data['year']:
                                col = f"{category}_{metric}_{stat}_{year}"
                                value = cat_data[cat_data['year'] == year][f"{metric}_{stat}"].iloc[0] if not cat_data.empty else 0
                                combined_df[col,:] = value
            if not extremes_df.empty:
                combined_df = pd.concat([combined_df, extremes_df], axis=1)

            combined_output_path = output_path.parent / 'combined_metrics_analysis.csv'
            combined_df.to_csv(combined_output_path, index=False)
            logger.info(f"Saved CSV to {combined_output_path}")


            # Generate HTML report
            env = Environment(loader=FileSystemLoader(str(current_dir / 'templates')))
            try:
                template = env.get_template('report_template.html')
            except TemplateNotFound:
                logger.error("HTML template not found")
                raise FileOperationError("HTML template not found")
            
            def style_total(df):
                return df.style.apply(lambda x: ['background-color: #d4eaff; font-weight: bold' if x['year'] == 'Total' else '' for _ in x], axis=1)

            extremes_all_total = style_total(extremes_df[extremes_df['scope'] == 'all_total']).to_html(index=True, index_names=['Scope', 'Year', 'Category']) if not extremes_df[extremes_df['scope'] == 'all_total'].empty else "<p>No all total extremes data</p>"
            extremes_category_total = style_total(extremes_df[extremes_df['scope'] == 'category_total']).to_html(index=True, index_names=['Scope', 'Year', 'Category']) if not extremes_df[extremes_df['scope'] == 'category_total'].empty else "<p>No category total extremes data</p>"
            extremes_all_yearly = extremes_df[extremes_df['scope'] == 'all_yearly'].to_html(index=True, index_names=['Scope', 'Year', 'Category']) if not extremes_df[extremes_df['scope'] == 'all_yearly'].empty else "<p>No all yearly extremes data</p>"
            extremes_category_yearly = extremes_df[extremes_df['scope'] == 'category_yearly'].to_html(index=True, index_names=['Scope', 'Year', 'Category']) if not extremes_df[extremes_df['scope'] == 'category_yearly'].empty else "<p>No category yearly extremes data</p>"

            html_report = template.render(
                date=time.strftime("%Y-%m-%d %H:%M:%S"),
                backtest_folder=output_path.parent.name,
                overall_table=overall_metrics_df.to_html(index=False, classes="table table-striped") if not overall_metrics_df.empty else "<p>No overall metrics data</p>",
                category_metrics_table=category_metrics_df.to_html(index=False, classes="table table-striped") if not category_metrics_df.empty else "<p>No category metrics data</p>",
                extremes_table_all_total=extremes_all_total,
                extremes_table_category_total=extremes_category_total,
                extremes_table_all_yearly=extremes_all_yearly,
                extremes_table_category_yearly=extremes_category_yearly,
            )
            html_output_path = output_path.parent / 'combined_metrics_report.html'
            with open(html_output_path, 'w') as f:
                f.write(html_report)
            logger.info(f"Saved HTML report to {html_output_path}")

            # Generate Markdown report
            try:
                template = env.get_template('report_template.md')
            except TemplateNotFound:
                logger.error("Markdown template not found")
                raise FileOperationError("Markdown template not found")

            extremes_all_total_md = extremes_df[extremes_df['scope'] == 'all_total'].to_markdown(index=True, tablefmt="grid") if not extremes_df[extremes_df['scope'] == 'all_total'].empty else "No all total extremes data"
            extremes_category_total_md = extremes_df[extremes_df['scope'] == 'category_total'].to_markdown(index=True, tablefmt="grid") if not extremes_df[extremes_df['scope'] == 'category_total'].empty else "No category total extremes data"
            extremes_all_yearly_md = extremes_df[extremes_df['scope'] == 'all_yearly'].to_markdown(index=True, tablefmt="grid") if not extremes_df[extremes_df['scope'] == 'all_yearly'].empty else "No all yearly extremes data"
            extremes_category_yearly_md = extremes_df[extremes_df['scope'] == 'category_yearly'].to_markdown(index=True, tablefmt="grid") if not extremes_df[extremes_df['scope'] == 'category_yearly'].empty else "No category yearly extremes data"

            markdown_report = template.render(
                date=time.strftime("%Y-%m-%d %H:%M:%S"),
                backtest_folder=output_path.parent.name,
                overall_table=overall_metrics_df.to_markdown(index=False) if not overall_metrics_df.empty else "No overall metrics data",
                category_metrics_table=category_metrics_df.to_markdown(index=False) if not category_metrics_df.empty else "No category metrics data",
                extremes_table_all_total=extremes_all_total_md,
                extremes_table_category_total=extremes_category_total_md,
                extremes_table_all_yearly=extremes_all_yearly_md,
                extremes_table_category_yearly=extremes_category_yearly_md,
            )
            markdown_output_path = output_path.parent / 'combined_metrics_report.md'
            with open(markdown_output_path, 'w') as f:
                f.write(markdown_report)
            logger.info(f"Saved Markdown report to {markdown_output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise FileOperationError(f"Failed to save results: {e}")

    def plot_category_distribution(self) -> None:
        """Plot distribution of alpha categories."""
        try:
            categories = ['pass', 'rerun', 'drop', 'not_run']
            counts = [len(self.transfer_lis), len(self.rerun_lis), len(self.drop_lis), len(self.run_lis)]
            plt.figure(figsize=(8, 6))
            plt.bar(categories, counts, color=['green', 'blue', 'red', 'gray'])
            plt.title('Alpha Category Distribution')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.grid(True, axis='y')
            plot_path = Path(self.config.out_path) / 'category_distribution.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"Saved category plot to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot category distribution: {e}")
            raise FileOperationError(f"Failed to plot: {e}")

    def fix_alpha_files(self) -> None:
        """Fix alpha_cls.py files for rerun alphas."""
        if not self.rerun_lis:
            logger.info("No rerun alphas to fix")
            return

        for alpha in tqdm(self.rerun_lis, desc="Fixing rerun alphas"):
            alpha_path = Path(backtest_path) / self.config.backtest_folder / alpha / 'alpha_cls.py'
            if not alpha_path.exists():
                logger.warning(f"alpha_cls.py not found for {alpha}")
                continue

            try:
                backup_path = backup_alpha(alpha_path)
                with open(alpha_path, 'r') as f:
                    content = f.read()
                tvr_max = getattr(self.config, 'tvr_max', 0.6) * 0.9
                tvr_min = getattr(self.config, 'tvr_min', 0.15) * 1.1
                new_content = re.sub(r'tvr_max\s*=\s*[\d.]+', f'tvr_max = {tvr_max:.3f}', content)
                new_content = re.sub(r'tvr_min\s*=\s*[\d.]+', f'tvr_min = {tvr_min:.3f}', new_content)
                if new_content != content:
                    with open(alpha_path, 'w') as f:
                        f.write(new_content)
                    logger.info(f"Updated {alpha} with tvr_max={tvr_max:.3f}, tvr_min={tvr_min:.3f}")
            except Exception as e:
                logger.error(f"Failed to fix {alpha}: {e}")
                continue

    def trans_test_compare(self) -> None:
        """Transfer or update passing alphas."""
        if not self.transfer_lis:
            logger.info("No passing alphas to transfer")
            return

        target_dir = Path(update_alpha_path) / self.config.backtest_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        for alpha in tqdm(self.transfer_lis, desc="Transferring alphas"):
            source_path = Path(backtest_path) / self.config.backtest_folder / alpha
            target_path = target_dir / alpha
            try:
                move_alpha_directory(source_path, target_path)
                comparison_result = compare_alpha_cls(str(target_path / 'alpha_cls.py'), str(source_path / 'alpha_cls.py'))
                update_total_alpha(alpha, str(target_path))
                logger.info(f"Transferred {alpha} and updated total_alpha")
            except Exception as e:
                logger.error(f"Failed to transfer {alpha}: {e}")
                continue