from Pisa import best_partition
from collections import defaultdict

import networkx as nx
g = nx.karate_club_graph()
club = nx.get_node_attributes(g, 'club')
part, com_labels = best_partition(g)

coms = defaultdict(int)
for n, c in part.items():
    coms[c] += 1

print(dict(coms))
print(com_labels)