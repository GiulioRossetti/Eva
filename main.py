from Pisa import best_partition, modularity, purity
from collections import defaultdict

import networkx as nx
g = nx.karate_club_graph()
club = nx.get_node_attributes(g, 'club')


for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    part, com_labels = best_partition(g, alpha=alpha, beta=1-alpha)

    print("\n")
    print(f"{alpha}\{1-alpha}")

    coms = defaultdict(int)
    for n, c in part.items():
        coms[c] += 1

    print(dict(coms))
    print(com_labels)

    print(f"{modularity(part, g)} - {purity(com_labels)}")
