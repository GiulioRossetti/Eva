# -*- coding: utf-8 -*-
"""
This module implements Pisa detection.
"""
from __future__ import print_function

import numbers
import warnings
from copy import deepcopy
import networkx as nx
import numpy as np
from .status import Status
import operator
__all__ = ["eva_best_partition", "purity", "modularity", "generate_dendrogram", "induced_graph",
           "eva_partition_at_level"]

__author__ = """Louvain algorithm: Thomas Aynaud (thomas.aynaud@lip6.fr)\n
                Eva extension: Giulio Rossetti (giulio.rossetti@isti.cnr.it), Salvatore Citraro"""

__PASS_MAX = -1
__MIN = 0.0000001


def __check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def eva_partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]

    Returns
    -------
    partition : dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to

    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high

    See Also
    --------
    best_partition which directly combines partition_at_level and
    generate_dendrogram to obtain the partition of highest modularity

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", eva_partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    """Compute the modularity of a partition of a graph

    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    modularity : float
       The modularity

    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating Pisa
    structure in networks. Physical Review E 69, 26113(2004).

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> part = eva_best_partition(G)
    >>> modularity(part, G)
    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def eva_best_partition(graph,
                       partition=None,
                       weight='weight',
                       resolution=1.,
                       randomize=None,
                       random_state=None,
                       alpha=0.5,
                       hierarchies=None):
    """Compute the partition of the graph nodes which maximises the modularity
    and label purity

    Parameters
    ----------
    graph : networkx.Graph
       the networkx graph which is decomposed
    partition : dict, optional
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize : boolean, optional
        Will randomize the node evaluation order and the Pisa evaluation
        order to get different partitions at each call
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    alpha : float
        The weight to give to the purity component. The value must lie in [0,1]

    Returns
    -------
    partition : dictionnary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not Eulerian.

    See Also
    --------
    generate_dendrogram to obtain all the decompositions levels


    Examples
    --------
    >>>  #Basic usage
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> part = eva_best_partition(G)
    """

    if 1 <= alpha <= 0:
        raise ValueError("Alpha must be positive floating point numbers in [0,1]")

    dendo, labels = generate_dendrogram(graph,
                                        partition,
                                        weight,
                                        resolution,
                                        randomize,
                                        random_state,
                                        alpha,
                                        hierarchies
                                        )
    return eva_partition_at_level(dendo, len(dendo) - 1), labels


def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None,
                        alpha=0.5,
                        hierarchies=None
                        ):
    """Find communities in the graph and return the associated dendrogram

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger
    are the communities


    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed
    part_init : dict, optional
        the algorithm will start using this partition of the nodes. It's a
        dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona

    Returns
    -------
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph

    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph

    See Also
    --------
    best_partition

    :param random_state:
    :param randomize:
    :param resolution:
    :param part_init:
    :param alpha:
    :param weight:
    :type weight:
    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = __check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its Pisa
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    __one_level(current_graph, status, weight, resolution, random_state, alpha, hierarchies)
    new_mod = __modularity(status)
    new_purity = __overall_purity(status)

    partition, status = __renumber(status.node2com, status)
    status_list.append(partition)
    mod = new_mod
    pur = new_purity
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_state, alpha, hierarchies)
        new_mod = __modularity(status)
        new_purity = __overall_purity(status)
        score = alpha * (new_purity - pur) + (1 - alpha) * (new_mod - mod)

        if score < __MIN:
            partition, status = __renumber(status.node2com, status)
            status_list.append(partition)
            break
        partition, status = __renumber(status.node2com, status)
        status_list.append(partition)
        mod = new_mod
        pur = new_purity
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)

    return status_list[:], status.com_attr


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    Parameters
    ----------
    partition : dict
       a dictionary where keys are graph nodes and  values the part the node
       belongs to
    graph : networkx.Graph
        the initial graph
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    g : networkx.Graph
       a networkx graph where nodes are the parts
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary, status):
    """Renumber the values of the dictionary from 0 to n
    """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])

    old_com_2_new_com = {}

    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value

        if value not in old_com_2_new_com:
            old_com_2_new_com[value] = new_value

    temp = dict([])

    for k in set(dictionary.values()):
        com_id = old_com_2_new_com[k]
        temp[com_id] = status.com_attr[k]

    status.com_attr = temp

    return ret, status


def __one_level(graph, status, weight_key, resolution, random_state, alpha, hierarchies=None):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status)
    new_mod = cur_mod
    curr_purity = __overall_purity(status)
    new_purity = curr_purity

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        curr_purity = new_purity

        modified = False
        nb_pass_done += 1

        for node in random_state.permutation(list(graph.nodes())):
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = - resolution * neigh_communities.get(com_node, 0) + \
                          (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)

            best_com = com_node
            best_increase = 0
            best_size_incr = 0
            n_com = len(neigh_communities)
            ti = 0
            for com, dnc in random_state.permutation(list(neigh_communities.items())):

                # increase cost label, gain community size
                initial_labels = status.attr[com_node]
                incr_labels = status.com_attr[com]
                incr_attr, incr_size = __delta_purity_size(initial_labels, incr_labels, hierarchies)

                # increase cost modularity
                incr = remove_cost + resolution * dnc - status.degrees.get(com, 0.) * degc_totw

                total_incr = alpha * incr_attr + (1 - alpha) * incr

                # check for increase in quality or in community size (with stable quality)
                if total_incr > best_increase or (total_incr == best_increase and incr_size > best_size_incr):
                    best_increase = total_incr
                    best_size_incr = incr_size
                    best_com = com
                else:
                    ti += 1
                    if ti == n_com:
                        break
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True

        new_mod = __modularity(status)
        new_purity = __overall_purity(status)

        score = alpha * (new_purity - curr_purity) + (1 - alpha) * (new_mod - cur_mod)

        if score < __MIN:
            break


def __delta_purity_size(original_attr1, new_attr2, hierarchies=None):
    total_original = 0
    for k, lst in original_attr1.items():
        for _, v in lst.items():
            total_original += v

    dumping = 1
    if hierarchies is not None:
        for attr in original_attr1:
            most_freq_label=max(original_attr1[attr].items(), key=operator.itemgetter(1))[0]
            most_freq_label_new=max(new_attr2[attr].items(), key=operator.itemgetter(1))[0]
            d = __distance(most_freq_label, most_freq_label_new, hierarchies[attr])
            dumping *= d

    purity_original = np.prod([(max(x / sum(y.values()) for x in y.values())) for y in original_attr1.values()])

    total_original /= len(original_attr1)
    total_nodes = total_original

    # computing original purity

    new_nodes = 0
    for k, lst in new_attr2.items():
        for _, v in lst.items():
            new_nodes += v
    total_nodes += new_nodes / len(new_attr2)

    updated = deepcopy(original_attr1)

    for k, lst in new_attr2.items():

        if k in updated:
            for t, v in lst.items():
                if t in updated[k]:
                    updated[k][t] += v
                else:
                    updated[k][t] = v
        else:
            updated[k] = lst

    purity_overall = np.prod([(max(x / sum(y.values()) for x in y.values())) for y in updated.values()]) * dumping

    increment = purity_overall - purity_original
    delta_size = (total_nodes - total_original) / total_original

    return increment, delta_size


def __overall_purity(status):
    purities = []
    for _, lst in status.com_attr.items():
        com_pur = []
        for _, v in lst.items():
            tot = sum(v.values())
            if tot == 0:
                continue
            scores = []

            for k in v.values():
                scores.append(k / tot)

            com_pur.append(max(scores))
        purities.append(np.prod(com_pur))

    return np.mean(purities)


def purity(com_attr):
    st = Status()
    st.com_attr = deepcopy(com_attr)
    return __overall_purity(st)


def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status):
    """ Remove node from Pisa com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1

    for v in status.attr[node].keys():
        label = v
        if label in status.com_attr[com]:
            for k, vn in status.attr[node][label].items():
                if vn > 0:
                    status.com_attr[com][label][k] -= status.attr[node][label][k]


def __insert(node, com, weight, status):
    """ Insert node into Pisa and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))

    for v in status.attr[node].keys():
        label = v
        if label in status.com_attr[com]:

            for k in status.attr[node][label]:
                if k in status.com_attr[com][label]:
                    status.com_attr[com][label][k] += status.attr[node][label][k]
                else:
                    status.com_attr[com][label][k] = status.attr[node][label][k]
        else:
            status.com_attr[com][label] = status.attr[node][label]

    ###################
    from collections import defaultdict
    coms = defaultdict(int)
    for n, c in status.node2com.items():
        coms[c] += 1


def __modularity(status):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree / links - ((degree / (2. * links)) ** 2)
    return result


def __distance(value1, value2, hierarchy):
    d = 1 - abs(hierarchy[value1] - hierarchy[value2])/(len(hierarchy))
    return d
