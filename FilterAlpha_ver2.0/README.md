Backtest File Processing Package
This package provides tools for analyzing, filtering, transferring, and categorizing backtest alpha files. It is designed for processing financial backtest data, particularly for alpha strategies.
Installation

Clone the repository or download the package files.
Create a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt



# Modules

config_validation: Validates configuration parameters using Pydantic. Supports YAML file loading.

file_operations: Handles file and directory operations (move, delete, compare, backup).

alpha_checks: Performs threshold checks on alpha metrics (IC, TVR, Drawdown, etc.).

filtering: Filters and categorizes alpha files based on predefined rules.

data_reading: Reads and processes backtest stats files (CSV format).


# Usage
1. Example: Filtering Alphas
```
from backtest_processing.filtering import FilterAlpha
from backtest_processing.config_validation import FilterAlphaConfig

config = {
    'start_date': '2020-01-01',
    'ic': 0.1,
    'icir': 0.5,
    'dd': 0.2,
    'win': 0.6,
    'ret': 0.05,
    'ret_l': 0.03,
    'tvr_max': 0.8,
    'tvr_min': 0.2,
    'sharpe': 1.0,
    'cover': 0.9,
    'out_path': 'output'
}

filter_alpha = FilterAlpha(config)
alphas, stats_paths = filter_alpha.run_backtest_file('signals_16')
filter_alpha.run_filter_backtest_average(alphas, stats_paths)
filter_alpha.trans_test_compare()
filter_alpha.rmtree_drop_alpha()
```

2. Example: Loading Config from YAML
config.yaml

```
start_date: '2020-01-01'
ic: 0.1
icir: 0.5
dd: 0.2
win: 0.6
ret: 0.05
ret_l: 0.03
tvr_max: 0.8
tvr_min: 0.2
sharpe: 1.0
cover: 0.9
out_path: 'output'

from backtest_processing.config_validation import load_config
from backtest_processing.filtering import FilterAlpha

config = load_config('config.yaml')
filter_alpha = FilterAlpha(config)
# Proceed with filtering as above
```
# Directory Structure

```
backtest_processing/
├── __init__.py
├── config_validation.py
├── file_operations.py
├── alpha_checks.py
├── filtering.py
├── data_reading.py
├── path_set.py
├── requirements.txt
└── README.md
```


## Testing
Unit tests can be added in a tests/ directory using pytest. Example:
``` {bash}
pip install pytest
pytest tests/
```
Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.
