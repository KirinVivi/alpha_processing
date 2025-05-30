## 📁 项目结构说明


```
alpha_processing/
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
├── README.md                     
├── pyproject.toml                

```