# Eva - Community Discovery for Labeled graphs.

[![pyversions](https://img.shields.io/pypi/pyversions/eva_lcd.svg)](https://badge.fury.io/py/eva_lcd)
[![PyPI version](https://badge.fury.io/py/eva_lcd.svg)](https://badge.fury.io/py/eva_lcd)
[![Build Status](https://travis-ci.org/GiulioRossetti/Eva.svg?branch=master)](https://travis-ci.org/GiulioRossetti/Eva)
[![Coverage Status](https://coveralls.io/repos/github/GiulioRossetti/Eva/badge.svg?branch=master)](https://coveralls.io/github/GiulioRossetti/Eva?branch=master)
[![DOI](https://zenodo.org/badge/191769795.svg)](https://zenodo.org/badge/latestdoi/191769795)
[![PyPI download month](https://img.shields.io/pypi/dm/eva_lcd.svg?color=blue&style=plastic)](https://pypi.python.org/pypi/eva_lcd/)


## Citation
If you use our algorithm please cite the following works:

> Citraro S., Rossetti G. (2020) Eva: Attribute-Aware Network Segmentation. In: Cherifi H., Gaito S., Mendes J., Moro E., Rocha L. (eds) Complex Networks and Their Applications VIII. COMPLEX NETWORKS 2019. Studies in Computational Intelligence, vol 881. Springer, Cham. https://doi.org/10.1007/978-3-030-36687-2_12

> Citraro, S., Rossetti, G. Identifying and exploiting homogeneous communities in labeled networks. Appl Netw Sci 5, 55 (2020). https://doi.org/10.1007/s41109-020-00302-1


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

An example of execution is also showed in the file "EVA_Example.ipynb"
