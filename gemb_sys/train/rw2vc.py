from __future__ import print_function
import random
import math
import numpy as np
import time

from vctrainer import vctrainer

class rw2vc(vctrainer):
    def __init__(self,
                 graph, rw_file, window, emb_file,
                 emb_model,
                 rep_size=128, epoch = 10, batch_size=1000, learning_rate=0.001, negative_ratio=5):
        start = time.time()
        self.window = window
        self.negative_ratio = negative_ratio
        self.batches = self.load_random_walks(rw_file)
        print("preprocessing time: {!s} for total pairs {}".format(time.time()-start, len(self.batches[0])))
        super(rw2vc, self).__init__(graph, None, None, emb_file=emb_file,
                                    emb_model=emb_model, rep_size=rep_size, epoch=epoch, batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    negative_ratio=negative_ratio)

    def train_one_epoch(self):
        sum_loss = 0.0

        batch_id = 0

        tot_time = 0.0
        start = time.time()
        for batch in self.batch_iter():
            h1, t1, sign = batch
            tx = time.time()
            # cur_loss = 0.0
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                self.h: h1, self.t: t1, self.sign: sign})
            tot_time += time.time() - tx
            sum_loss += cur_loss
            batch_id += 1
            # if batch_id % 10 == 0:
            #    print("per batch costs {}".format(tot_time/batch_id))
        end = time.time()

        print('epoch {}: sum of loss:{!s}; time cost: {!s}/{!s}, per_batch_cost: {!s}'.
              format(self.cur_epoch, sum_loss / batch_id, tot_time, end - start, tot_time / batch_id))

    def batch_iter(self):
        h0, t0, sign0 = self.batches
        shuffle_indices = np.arange(len(h0)) #np.random.permutation(np.arange(len(h0)))

        table_size = len(shuffle_indices)
        i = 0
        while i < table_size:
            h = []
            t = []
            sign = []
            j = min(i + self.batch_size, table_size)
            # print(i,j)
            for k in range(i, j):
                idx = shuffle_indices[k]
                h.append(h0[idx])
                t.append(t0[idx])
                sign.append(1.0)
                for negId in range(self.negative_ratio):
                    h.append(h0[idx])
                    t.append(random.randint(0, self.node_size -1))
                    sign.append(0.0)
            yield h, t, sign
            i = j

        # for walk in walks:
        #    cnt = cnt + 1
        #    if cnt % 10000 == 0:
        #     print("Progress: {}/{}".format(cnt, len(walks)))
        #    for i in range(len(walk)):
        #        for j in range(i-self.window, i+self.window + 1):
        #            if j > -1 and j != i and j < len(walk):
        #                sample_count = sample_count + 1
        #                h.append(walk[i])
        #                t.append(walk[j])
        #                sign.append(1.0)
        #                for k in range(self.negative_ratio):
        #                    h.append(walk[i])
        #                    t.append(random.randint(0, self.node_size - 1))
        #                    sign.append(0.0)
        #                if sample_count >= self.batch_size:
        #                    yield h, t, sign
        #                    sample_count = 0
        #                    h = []
        #                    t = []
        #                    sign = []
        # if len(h) > 0:
        #     yield h, t, sign

    def load_random_walks(self, rw_file):
        h, t, sign = [], [], []
        with open(rw_file, 'r') as f:
            for line in f:
                walk = line.split()
                for i in range(len(walk)):
                    for j in range(i - self.window, i + self.window + 1):
                        if j > -1 and j != i and j < len(walk):
                            h.append(walk[i])
                            t.append(walk[j])
                            sign.append(1.0)
        return h, t, sign
