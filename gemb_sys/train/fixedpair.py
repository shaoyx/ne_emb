from __future__ import print_function
import math
from graph import *

class fixedpair(object):
    def __init__(self, graph, pair_file):
        self.g = graph
        if graph.directed:
            self.g_r = None
        else:
            self.g_r = graph
        self.pair_file = pair_file

    def sample_batch(self, batch_size):
        batches = []
        with open(self.pair_file, "r") as f:
            for line in f:
                rec = line.split()
                batches.append(rec)
        random.shuffle(batches)

        # cnt = 0;
        h, t = [], []
        for batch in batches:
            h.append(batch[0])
            t.append(batch[1])
            if len(h) == batch_size:
                # cnt = cnt + 1
                # if cnt % 10 == 0:
                #     print(cnt)
                yield h,t
                h, t = [], []
        if len(h) > 0:
            yield h, t
