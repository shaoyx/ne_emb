from __future__ import print_function
import time
import numpy as np
import random
import math
from .graph import *

class deepwalk(object):
    # fac*node_size is the size of v_sampling table (for each epoch)
    def __init__(self, graph, fac=50, window=10):
        self.g = graph
        if graph.directed:
            self.g_r = None
        else:
            self.g_r = graph
        self.fac = fac
        self.window = window
        self.app = None
        self.it = None

    def sample_v(self, batch_size):
        try:
            nodes = self.nodes
        except:
            self.nodes = list(self.g.G.nodes())
            nodes = self.nodes
        numNodes = len(nodes)
        table_size = self.fac * numNodes

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes)

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]
                        ] += self.g.G[edge[0]][edge[1]]["weight"]
        norm = sum([node_degree[i] for i in range(numNodes)])

        # sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        h = []
        for j in range(numNodes):
            p += float(node_degree[j]) / norm
            while i < table_size and float(i) / table_size < p:
                h += [j]
                i += 1
                if i % batch_size == 0:
                    yield h
                    h = []
        if len(h) > 0:
            yield h

    def sample_c(self, h):
        if self.it is None:
            self.it = {} # how many steps
            for i in self.g.G.nodes():
                self.it[i] = 0
            self.pl = {} # where it stopped
            self.neighbors = {}
            self.degrees = {}
            for root in self.g.G.nodes():
                self.neighbors[root] = list(self.g.G.neighbors(root))
                self.degrees[root] = len(self.neighbors[root])
        degrees = self.degrees
        neighbors = self.neighbors
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        i = 0
        batch_size = len(h)
        t = []
        while i < batch_size:
            root = look_back[h[i]]
            if degrees[root] == 0:
                t += [look_up[root]]
                i += 1
                continue
            if self.it[root] == 0:
                iid = root
                self.it[root] = self.window
            else:
                iid = self.pl[root]
            if degrees[iid] == 0:
                iid = root
                self.it[root] = self.window
            iid = random.choice(neighbors[iid])
            self.it[root] -= 1
            self.pl[root] = iid
            t += [look_up[iid]]
            i += 1
        return t

