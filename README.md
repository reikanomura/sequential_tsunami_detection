# sequential_tsunami_detection

Python codes for the real-time tsunami scenario detection.

- keywords
  - Bayesian update
  - Singular Value Docomposition (SVD), Proper Orthogonal Decomposition (POD)


```
┌── data
│   ├── obs_pnts.txt       Input file: Synthetic gauges used in the running
│   └── cases.txt          Input file: Scenarios used in the running
├── pgm
│   ├── run2.py            Main code
│   ├── ttsplit.py         Subroutine: Test/Training data splitting
│   ├── POD.py             Subroutine: Proper orthogonal decomposition
│   ├── psudo_inv.py       Subroutine: Pseudo inverse 
│   ├── bayesian_update.py Subroutine: Bayesian update
│   ├── beautyfun.py       Subroutine: Various functions for data handling
│   └── graphing.py        Subroutine: For graph making
└── res                    Output directory
```

## Environment

We confirmed the code running under the following environment.
- CentOS(Linux) Ver.7
- python 3.8.7 (via pyenv)
- TORQUE resource manager


## Required python libraries

- numpy==1.21.2
- matplotlib==3.4.3
- scipy==1.7.1
- dask==2021.8.1
- seaborn==0.11.2


## References

[![DOI](https://zenodo.org/badge/508964446.svg)](https://zenodo.org/badge/latestdoi/508964446)

