#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package implements Pisa detection.

Package name is Pisa but refer to python-louvain on pypi
"""

from .pisa import (
    partition_at_level,
    modularity,
    best_partition,
    generate_dendrogram,
    induced_graph,
    load_binary,
)

__version__ = "0.13"
__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.
