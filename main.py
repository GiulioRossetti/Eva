from Pisa import best_partition, modularity, purity
from collections import defaultdict
import community as louvain

import networkx as nx
import random

#g = nx.karate_club_graph()
#club = nx.get_node_attributes(g, 'club')

labels = ['one', 'two', 'three', 'four']
age = ["A", "B", "C"]
g =nx.barabasi_albert_graph(100, 5)

for node in g.nodes():
    g.add_node(node, labels=random.choice(labels), age=random.choice(age))


cms = louvain.best_partition(g)
mods = louvain.modularity(cms, g)
print(f"Original Louvain Modularity: {mods}")


for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    part, com_labels = best_partition(g, alpha=alpha)

    print("\n")
    print(f"{alpha}\{1-alpha}")

    coms = defaultdict(int)
    for n, c in part.items():
        coms[c] += 1

    print(sorted(dict(coms).items(), key=lambda x: x[0]))
    print(sorted(com_labels.items(), key=lambda x: x[0]))

    for k, v in coms.items():
        l_count = 0
        for l, lst in com_labels[k].items():
            # print(l, lst)
            for _, v in lst.items():
                l_count += v
        l_count /= len(com_labels[k].items())
        # print(k, coms[k], l_count)
        assert coms[k] == l_count

    print(f"{modularity(part, g)} - {purity(com_labels)}")

