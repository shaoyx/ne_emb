from __future__ import print_function
import random
import numpy as np
import time

import tensorflow as tf


class rw2vc(object):
    def __init__(self,
                 graph, rw_file, window, emb_file,
                 emb_model,
                 rep_size=128, epoch = 10, batch_size=1000, learning_rate=0.001, negative_ratio=5):

        self.g = graph
        self.emb_model = emb_model  # sym (first), asym (second)
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.emb_file = emb_file

        start = time.time()
        self.window = window
        self.walks = self.load_random_walks(rw_file)
        print("preprocessing time: {!s} for total walks {}".format(time.time()-start, len(self.walks)))

        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=cur_seed)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_model(self.emb_model)

        self.sess.run(tf.global_variables_initializer())
        print("Start training.")
        self.cur_epoch = 0
        for i in range(epoch):
            self.train_one_epoch()
            self.cur_epoch += 1
            if self.cur_epoch % 5 == 0:
                self.get_embeddings()
                self.save_embeddings(self.emb_file + "_" + str(self.cur_epoch))
        self.get_embeddings()

    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.eval(session=self.sess)
        # embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        self.vectors = vectors


    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()


    def build_model(self, emb_model):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])

        cur_seed = random.getrandbits(32)
        if emb_model == 'asym':
            print("using asym loss!")
            self.embeddings = tf.get_variable(name="embeddings", shape=[
                                          self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
            self.context_embeddings = tf.get_variable(name="context_embeddings", shape=[
                                                  self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
            self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
            self.t_e = tf.nn.embedding_lookup(self.context_embeddings, self.t) #context emb, second order loss
        else:
            print("using sym loss!")
            self.embeddings = tf.get_variable(name="embeddings", shape=[
                self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
            self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
            self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t) #word_emb, first order loss

        # self.loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(
        #     self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)),1e-8,1.0))) # why use clip?
        # self.loss = -tf.reduce_mean(tf.log_sigmoid(
        #     self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
        logits = tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.sign))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def neg_batch(self, h):
        t = []
        for i in range(len(h)):
            t.append(random.randint(0, self.node_size-1))
        return t

    def train_one_epoch(self):
        sum_loss = 0.0
        batch_id = 0
        tot_time = 0.0
        start = time.time()
        for batch in self.generate_batch():
            h1, t1 = batch
            sign = [1.0 for _ in range(len(h1))]
            hx = [x for x in h1]
            for i in range(self.negative_ratio):  # The training order matters!!!!
                t_neg = self.neg_batch(h1)
                for idx in range(len(h1)):
                    t1.append(t_neg[idx])
                    hx.append(h1[idx])
                    sign.append(0.0)

            tx = time.time()
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                self.h: hx, self.t: t1, self.sign: sign})
            tot_time += time.time() - tx
            sum_loss += cur_loss
            batch_id += 1
        end = time.time()

        print('epoch {}: sum of loss:{!s}; time cost: {!s}/{!s}, per_batch_cost: {!s}'.
              format(self.cur_epoch, sum_loss / batch_id, tot_time, end - start, tot_time / batch_id))

    def generate_batch(self):
        shuffle_indices = np.random.permutation(np.arange(len(self.walks)))

        # table_size = len(shuffle_indices)
        # i = 0
        # while i < table_size:
        #     h = []
        #     t = []
        #     sign = []
        #     j = min(i + self.batch_size, table_size)
        #     # print(i,j)
        #     for k in range(i, j):
        #         idx = shuffle_indices[k]
        #         h.append(h0[idx])
        #         t.append(t0[idx])
        #         sign.append(1.0)
        #         for negId in range(self.negative_ratio):
        #             h.append(h0[idx])
        #             t.append(random.randint(0, self.node_size -1))
        #             sign.append(0.0)
        #     yield h, t
        #     i = j

        h, t = [], []
        cnt = 0
        for idx in shuffle_indices:
            walk = self.walks[idx]
            for i in range(len(walk)):
                for j in range(i-self.window, i+self.window + 1):
                    if j > -1 and j != i and j < len(walk):
                        # sample_count = sample_count + 1
                        h.append(walk[i])
                        t.append(walk[j])
                        cnt = cnt + 1
                        if len(h) >= self.batch_size:
                            yield h, t
                            cnt = 0
                            h = []
                            t = []
        if len(h) > 0:
            yield h, t

    def load_random_walks(self, rw_file):
        walks = []
        with open(rw_file, 'r') as f:
            for line in f:
                walk = line.split()
                walks.append(walk)
        return walks
        # h, t, sign = [], [], []
        # with open(rw_file, 'r') as f:
        #     for line in f:
        #         walk = line.split()
        #         for i in range(len(walk)):
        #             for j in range(i - self.window, i + self.window + 1):
        #                 if j > -1 and j != i and j < len(walk):
        #                     h.append(walk[i])
        #                     t.append(walk[j])
        #                     sign.append(1.0)
        # return h, t, sign
