# Eva - Community Discovery for Labeled graphs.

[![Build Status](https://travis-ci.org/GiulioRossetti/Eva.svg?branch=master)](https://travis-ci.org/GiulioRossetti/Eva)
[![Coverage Status](https://coveralls.io/repos/github/GiulioRossetti/Eva/badge.svg?branch=master)](https://coveralls.io/github/GiulioRossetti/Eva?branch=master)
[![pyversions](https://img.shields.io/pypi/pyversions/eva.svg)](https://badge.fury.io/py/Eva)
[![PyPI version](https://badge.fury.io/py/eva.svg)](https://badge.fury.io/py/Eva)
[![Updates](https://pyup.io/repos/github/GiulioRossetti/Eva/shield.svg)](https://pyup.io/repos/github/GiulioRossetti/Eva/)


## Citation
If you use our algorithm please cite the following works:

## Installation


In order to install the package just download (or clone) the current project and copy the demon folder in the root of your application.

Alternatively use pip:
```bash
sudo pip install eva
```

Eva is written in python and requires the following package to run:
- networkx
- numpy

# Execution

eVA can be executed specifying as input: 

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
