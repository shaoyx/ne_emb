from __future__ import print_function
import random
import math
import numpy as np
import tensorflow as tf

from vctrainer import vctrainer

class rw2vc(vctrainer):
    def __init__(self,
                 graph, rw_file, window,
                 emb_model,
                 rep_size=128, epoch = 10, batch_size=1000, learning_rate=0.001, negative_ratio=5):
        # super(vctrainer, self).__init__(graph, None, None, emb_model, rep_size, epoch, batch_size, learning_rate, negative_ratio)
        # self.rw_file = rw_file
        self.walks = self.load_random_walks(rw_file)
        self.window = window
        super(rw2vc, self).__init__(graph, None, None, emb_model, rep_size, epoch, batch_size, learning_rate, negative_ratio)

    def train_one_epoch(self):
        sum_loss = 0.0
        pos_vc = self.batch_iter() #generating positive batches from random walks
        batch_id = 0
        for vc in pos_vc:
            h, t, sign = vc
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                self.h: h, self.t: t, self.sign: sign})
            sum_loss += cur_loss
            batch_id += 1 #positive batch
            for i in range(self.negative_ratio):
                t = self.neg_batch(h)
                sign = [-1.]
                _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.h: h, self.t: t, self.sign: sign})
                sum_loss += cur_loss
                batch_id += 1

        print('epoch {}: sum of loss:{!s}'.format(self.cur_epoch, sum_loss / batch_id))

    def batch_iter(self):

        walks = self.walks #self.load_random_walks(self.rw_file)
        walks = np.random.permutation(walks)

        sample_count = 0
        h = []
        t = []
        walk_count=0
        tot= len(walks)
        for walk in walks:
           walk_count=walk_count+1
           # if walk_count % 10000 == 0:
               # print("progress: {}/{}".format(walk_count, tot))
           for i in range(len(walk)):
               for j in range(i-self.window, i+self.window + 1):
                   if j > 0 and j != i and j < len(walk):
                       sample_count = sample_count + 1
                       h.append(walk[i])
                       t.append(walk[j])
                       if sample_count >= self.batch_size:
                           yield h, t, [1.0]
                           sample_count = 0
                           h = []
                           t = []
        if len(h) > 0:
            yield h, t, [1.0]
            h = []
            t = []

    def load_random_walks(self, rw_file):
        walks = []
        with open(rw_file, 'r') as f:
            for line in f:
                walks.append(line.split())
        return walks
