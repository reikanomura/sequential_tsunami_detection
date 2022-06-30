# sequential_tsunami_detection

## 

```
┌── data
│   ├── obs_pnts.txt       Input file: Synthetic gauges used in the running
│   └── cases.txt          Input file: Scenarios used in the running
├── pgm
│   ├── run2.py            Main codes
│   ├── ttsplit.py         Subroutine: Test/Training data splitting
│   ├── POD.py             Subroutine: Proper orthogonal decomposition
│   ├── psudo_inv.py       Subroutine: Pseudo inverse 
│   ├── bayesian_update.py Subroutine: Bayesian update
│   ├── beautyfun.py       Subroutine: Various data handling functions
│   └── graphing.py        Subroutine: Graph 
└── res                    Output directory
```

## Required environment

We confirmed the code can run under the following environment.

- CentOS(Linux) Ver.7
- python 3.8.7 (via pyenv)
  - numpy==1.21.2
  - matplotlib==3.4.3
  - scipy==1.7.1
  - dask==2021.8.1
  - seaborn==0.11.2

