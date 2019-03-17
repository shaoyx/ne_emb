from __future__ import print_function
import math
from graph import *

class generalwalk(object):
    # fac*node_size is the size of v_sampling table (for each epoch)
    def __init__(self, graph, fac=50, window=10, degree_bound=0, degree_power=1.0, alpha=0.15):
        self.g = graph
        if graph.directed:
            self.g_r = None
        else:
            self.g_r = graph
        self.fac = fac
        self.window = window
        self.degree_bound = degree_bound
        self.degree_power = degree_power
        self.alpha = alpha
        self.app = None
        self.it = None

    def sample_batch(self, batch_size):
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
        degree_bound = self.degree_bound
        if degree_bound > 0:
            for i in range(numNodes):
                if node_degree[i] > degree_bound:
                    node_degree[i] = degree_bound

        for i in range(numNodes):
            node_degree[i] = math.pow(node_degree[i], self.degree_power)

        norm = sum([node_degree[i] for i in range(numNodes)])

        sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        h = []
        for j in range(numNodes):
            p += float(node_degree[j]) / norm
            while i < table_size and float(i) / table_size < p:
                sampling_table[i] = j
                i += 1
        random.shuffle(sampling_table)
        i = 0
        while i < table_size:
            j = min(i + batch_size, table_size)
            h = sampling_table[i:j]
            t = self.sample_c_asym(h)
            yield h, t
            i = j

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
        degree_bound = self.degree_bound
        if degree_bound > 0:
            for i in range(numNodes):
                if node_degree[i] > degree_bound:
                    node_degree[i] = degree_bound

        for i in range(numNodes):
            node_degree[i] = math.pow(node_degree[i], self.degree_power)

        norm = sum([node_degree[i] for i in range(numNodes)])

        sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        h = []
        for j in range(numNodes):
            p += float(node_degree[j]) / norm
            while i < table_size and float(i) / table_size < p:
                sampling_table[i] = j
                i += 1
        random.shuffle(sampling_table)
        i = 0
        while i < table_size:
            j = min(i + batch_size, table_size)
            yield sampling_table[i:j]
            i = j

    def sample_c(self, h):
        # return self.sample_c_asym(h)
        return self.sample_c_sym(h)

    def sample_c_asym(self, h):
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
            try:
                root = look_back[h[i]]
            except:
                print(h[i])
                exit()
            if degrees[root] == 0: #skip the zero degree nodes.
                t += [look_up[root]]
                i += 1
                continue
            while 1:
                if self.it[root] == 0: #restrat
                    iid = root
                    self.it[root] = self.window
                else:
                    iid = self.pl[root] #continue the last node
                if degrees[iid] == 0: #skip the zero degree nodes
                    iid = root
                    self.it[root] = self.window
                iid = random.choice(neighbors[iid])
                self.it[root] -= 1
                self.pl[root] = iid
                prob = random.random()
                if prob < self.alpha: #output a node
                    break
            t += [look_up[iid]]
            i += 1
        return t

    def sample_c_sym(self, h):
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
        new_h = []
        t = []
        while i < batch_size:
            try:
                root = look_back[h[i]]
            except:
                print(h[i])
                exit()
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

            new_h += [h[i]]
            t += [look_up[iid]]

            new_h += [look_up[iid]]
            t += [h[i]]
            i += 1
        return new_h, t

