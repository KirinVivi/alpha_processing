# About Calculators

In this context, **calculators** refer to Python classes that encapsulate various data processing and transformation functions for quantitative alpha factors. They provide a unified interface for applying mathematical, statistical, and rolling window operations to numpy arrays or pandas DataFrames.

## Key Points

- **ProcessCalculator**:  
  Acts as a base class, offering function dispatch, validation, and shape alignment utilities. It allows flexible and dynamic application of processing functions to input data.

- **ProcessCalculatorL1**:  
  Extends the base calculator with a collection of Level-1 (basic) processing methods, such as normalization, ranking, rolling statistics, and outlier handling.  
  Each method is designed to be robust against `inf` and `nan` values, ensuring reliable results in quantitative research workflows.

- **Decorator Usage**:  
  Decorators like `fill_inf_with_max_min_ts_decorator` are used to automatically sanitize input and output data, further enhancing the reliability of all calculator methods.

## Purpose

These calculators are essential for:
- Standardizing data preprocessing steps in alpha factor research.
- Enabling reusable, modular, and maintainable code for quantitative analytics.
- Providing a foundation for higher-level feature engineering and signal processing.

---

**In summary:**  
Calculators in this folder are specialized classes for systematic, robust, and efficient transformation of financial factor data.

```
├── alpha_processing/             # main package
│   ├── calculators/              # alpha calculators module
│   │   ├── base_calculator.py    # basic 
│   │   ├── level1_calculator.py  # Level 1 : keep the data shape
│   │   ├── level2_calculator.py  # Level 2 : compress the data shape
│   ├── processors/               # alpha processors module
│   │   ├── level1_processor.py   # Level 1 
│   │   ├── level2_processor.py   # Level 2 
│   │   ├── np_cal_func.py        # vectorize func
│   │   ├── filter_func.py        # filter func
│   ├── utils/                    # utils func
│   │   ├── data_utils.py         # data loading and filling
│   ├── config.yaml               # config
│   ├── runs                      # run
|   |   ├── run_single.py 
|   |   ├── run_combine.py 
|   |   ├── run_single_delay.py 
|   |   ├── run_combine_delay.py 
|   |   ├── run_single_window.py 
|   |   ├── run_combine_window.py 
├── tests/                        # test
│   ├── test_calculators.py       # unit test：calculators
│   ├── test_utils.py             # unit test：processors

```