# Cllr and min Cllr computation
license: LGPLv3 <br/>
version: 2020-01-10 <br/>
<i>author: Andreas Nautsch (EURECOM)</i>

This standalone package is based on:
* pyBOSARIS <br/> https://gitlab.eurecom.fr/nautsch/pybosaris
* sidekit <br/> https://git-lium.univ-lemans.fr/Larcher/sidekit
* BOSARIS toolkit <br/> https://sites.google.com/site/bosaristoolkit/

In `performance.py`, one can find the derived code snippets.

## Installation
The installation uses miniconda, which creates Python environments into a folder structure on your hard drive. Deinstallation is easy: delete the miniconda folder.

1. install miniconda, see: <br/>
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/#regular-installation
2. create a Python environment
    >  conda create python=3.7 --name cllr -y
3. activate the environment
    > conda activate cllr
4. installing required packages
    > conda install -y numpy pandas

## How to compute the metrics
Computing the metrics (command structure):
```
python compute_cllr -s [SCORE_FILE] -k [KEY_FILE]
```

An example is provided with `scores.txt` and `key.txt` as score and key files:
> python compute_cllr scores.txt key.txt

which produces the output:
```
Cllr (min/act): 0.048/1.551
```

By using the flag `-e`, the ROCCH-EER will be computed, too:
```
ROCCH-EER: 1.4%
```

## Computation time
<i>Note: ROCCH-EER computation is optional to the computation of min Cllr.</i>

Measured on i7-8550U CPU @ 1.80GHz with 7 runs, 100 loops each:

Cllr: 790 µs ± 94.2 µs <br/>
min Cllr: 141 ms ± 3.48 ms <br/>
min Cllr & ROCCH-EER: 145 ms ± 1.49 ms
