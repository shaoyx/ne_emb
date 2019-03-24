from __future__ import print_function
from vcgenerator import *

"""
jump_factor: prob to stop
Paper: Scalable Graph Embedding for Asymmetric Proximity
"""


class APP(VCGenerator):

    def __init__(self, graph, batch_size=1000, stop_factor=0.15, sample=200, step=10):
        super(APP, self).__init__()
        self.g = graph
        self.nodes = graph.G.nodes()
        self.stop_factor = stop_factor
        self.sample_per_node = sample
        self.step = step
        self._pairs_per_epoch = int(self.g.G.number_of_nodes() * self.sample_per_node/batch_size) * batch_size
        self.neighbors = {}
        self.degrees = {}
        for root in self.nodes():
            self.neighbors[root] = list(graph.G.neighbors(root))
            self.degrees[root] = len(self.neighbors[root])

    def pair_per_epoch(self):
        return self._pairs_per_epoch

    def generate_batch(self, batch_size):
        shuffle_nodes = list(self.nodes)
        look_up = self.g.look_up_dict
        random.shuffle(shuffle_nodes)
        h = []
        cnt = 0
        totcnt = 0
        while True:
            for root in shuffle_nodes:
                cnt += 1
                totcnt = totcnt + 1
                h += [look_up[root]]
                if cnt >= batch_size:
                    t = self.sample_context(h)
                    yield h, t
                    cnt = 0
                    h = []
            if totcnt >= self._pairs_per_epoch:
                break
        # if len(h) > 0:
        #     t = self.sample_context(h)
        #     yield h, t

    def sample_context(self, h):
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        t = []
        for i in h:
            root = look_back[i]
            iid = root
            s = self.step
            while s > 0:
                s -= 1
                stop = random.random()
                if stop < self.stop_factor:
                    break
                if self.degrees[iid] == 0:
                    break
                iid = random.choice(self.neighbors[iid])
            t += [look_up[iid]]
        return t

    # def sample_v(self, batch_size):
    #     random.seed()
    #     try:
    #         nodes = self.nodes
    #     except:
    #         self.nodes = list(self.g.G.nodes())
    #         nodes = self.nodes
    #     look_up = self.g.look_up_dict
    #     random.shuffle(nodes)
    #     h = []
    #     cnt = 0
    #     for root in nodes:
    #         for i in range(self.sample):
    #             cnt += 1
    #             h += [look_up[root]]
    #             if cnt >= batch_size:
    #                 yield h
    #                 cnt = 0
    #                 h = []
    #     if len(h) > 0:
    #         yield h

    # def generate_a_batch(self, batch_size):
    #     random.seed()
    #     try:
    #         nodes = self.nodes
    #     except:
    #         self.nodes = list(self.g.G.nodes())
    #         nodes = self.nodes
    #     look_up = self.g.look_up_dict
    #     random.shuffle(nodes)
    #     hx = []
    #     cnt = 0
    #     for root in nodes:
    #         for i in range(self.sample):
    #             cnt += 1
    #             hx += [look_up[root]]
    #             if cnt >= batch_size:
    #                 tx = self.sample_c(hx)
    #                 h = []
    #                 t = []
    #                 sign = []
    #                 for idx in range(len(hx)):
    #                     h.append(hx[idx])
    #                     t.append(tx[idx])
    #                     sign.append(1.0)
    #                     for negId in range(negative_ratio):
    #                         h.append(hx[idx])
    #                         t.append(random.randint(0, self.g.G.number_of_nodes() - 1))
    #                         sign.append(0)
    #                 yield h, t, sign
    #                 cnt = 0
    #                 hx = []
    #     if len(hx) > 0:
    #         tx = self.sample_c(hx)
    #         h = []
    #         t = []
    #         sign = []
    #         for idx in range(len(hx)):
    #             h.append(hx[idx])
    #             t.append(tx[idx])
    #             sign.append(1.0)
    #             for negId in range(negative_ratio):
    #                 h.append(hx[idx])
    #                 t.append(random.randint(0, self.g.G.number_of_nodes() - 1))
    #                 sign.append(0)
    #         yield h, t, sign


    '''
    def batch_iter(self):
        random.seed()
        jump_factor = self.jump_factor
        sample = self.sample
        step = self.step
        batch_size = self.batch_size
        G = self.g.G
        if self.neighbors is None:
            self.neighbors = {}
            for root in G.nodes():
                self.neighbors[root] = list(G.neighbors(root))
        neighbors = self.neighbors
        look_up = self.g.look_up_dict
        try:
            nodes = self.nodes
        except:
            self.nodes = list(self.g.G.nodes())
            nodes = self.nodes
        random.shuffle(nodes)
        cnt = 0
        h = []
        t = []
        for root in nodes:
            cur_nbrs = neighbors[root]
            if len(cur_nbrs) == 0:
                continue
            for i in range(sample):
                s = step
                iid = -1
                while s > 0:
                    s -= 1
                    jump = random.random()
                    if jump < jump_factor:
                        break
                    iid = random.choice(cur_nbrs)
                    cur_nbrs = neighbors[iid]
                    if len(cur_nbrs) == 0:
                        break
                if iid != -1:
                    cnt += 1
                    h += [look_up[root]]
                    t += [look_up[iid]]
                    if cnt >= batch_size:
                        yield h, t, [1]
                        cnt = 0
                        h = []
                        t = []
                cur_nbrs = neighbors[root]
        if cnt > 0:
            yield h, t, [1]
    '''
