import unittest
from Eva import eva_best_partition, modularity, purity
from collections import defaultdict

import networkx as nx
import random


class EvaTestCase(unittest.TestCase):

    def test_eva(self):

        labels = ['one', 'two', 'three', 'four']
        age = ["A", "B", "C"]
        g = nx.barabasi_albert_graph(100, 5)

        for node in g.nodes():
            g.add_node(node, labels=random.choice(labels), age=random.choice(age))

        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            part, com_labels = eva_best_partition(g, alpha=alpha)

            coms = defaultdict(int)
            for n, c in part.items():
                coms[c] += 1

            for k, v in coms.items():
                l_count = 0
                for l, lst in com_labels[k].items():
                    for _, v in lst.items():
                        l_count += v
                l_count /= len(com_labels[k].items())
                self.assertEquals(coms[k], l_count)

            self.assertLessEqual(modularity(part, g), 1)
            self.assertLessEqual(purity(com_labels), 1)


if __name__ == '__main__':
    unittest.main()
