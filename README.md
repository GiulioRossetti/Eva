# Eva - Community Discovery for Labeled graphs.

[![pyversions](https://img.shields.io/pypi/pyversions/eva_lcd.svg)](https://badge.fury.io/py/eva_lcd)
[![PyPI version](https://badge.fury.io/py/eva_lcd.svg)](https://badge.fury.io/py/eva_lcd)
[![Build Status](https://travis-ci.org/GiulioRossetti/Eva.svg?branch=master)](https://travis-ci.org/GiulioRossetti/Eva)
[![Coverage Status](https://coveralls.io/repos/github/GiulioRossetti/Eva/badge.svg?branch=master)](https://coveralls.io/github/GiulioRossetti/Eva?branch=master)
[![DOI](https://zenodo.org/badge/191769795.svg)](https://zenodo.org/badge/latestdoi/191769795)
[![PyPI download month](https://img.shields.io/pypi/dm/eva_lcd.svg?color=blue&style=plastic)](https://pypi.python.org/pypi/eva_lcd/)


## Citation
If you use our algorithm please cite the following works:

> S. Citraro and G. Rossetti
>
> **Eva: Attribute-Aware Network Segmentation**
>
> Accepted to Complex Networks 2019
> arXiv preprint https://arxiv.org/abs/1910.06599

## Installation


In order to install the package just download (or clone) the current project and copy the demon folder in the root of your application.

Alternatively use pip:
```bash
sudo pip install eva_lcd
```

Eva is written in python and requires the following package to run:
- networkx
- numpy

# Execution

Eva can be executed specifying as input: 

- a (labeled) *networkx* Graph object
- the value of the trade-off parameter (alpha)

```python
import networkx as nx
from Eva import eva_best_partition, modularity, purity

g = nx.karate_club_graph()
part, com_labels = eva_best_partition(g, alpha=0.8)

q = modularity(part, g)
p = purity(com_labels)

```
