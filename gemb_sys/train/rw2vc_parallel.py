from __future__ import print_function
import random
import time
import threading
import math

import numpy as np
import tensorflow as tf

from vctrainer import vctrainer

"""
Too heavy to use the queue for parallel input!
"""
class rw2vc(vctrainer):
    def __init__(self,
                 graph, rw_file, window, emb_file,
                 emb_model,
                 rep_size=128, epoch = 10, batch_size=1000, learning_rate=0.001, negative_ratio=5, nthreads=8):
        start = time.time()
        self.window = window
        self.negative_ratio = negative_ratio
        self.batches = self.load_random_walks(rw_file)
        self.nthread = nthreads
        print("preprocessing time: {!s} for total pairs {}".format(time.time()-start, len(self.batches[0])))

        self.coord = tf.train.Coordinator()
        shuffle_indices = np.random.permutation(np.arange(len(self.batches[0])))
        self.reader = DataGenerator(self.batches, shuffle_indices, self.negative_ratio, graph.G.number_of_nodes(), self.coord,
                               max_queue_size=batch_size * 10)
        self.input_batch = self.reader.dequeue(batch_size)

        super(rw2vc, self).__init__(graph, None, None, emb_file=emb_file,
                                    emb_model=emb_model, rep_size=rep_size, epoch=epoch, batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    negative_ratio=negative_ratio)


    def build_model(self, emb_model):
        self.h = self.input_batch[0] #tf.placeholder(tf.int32, [None])
        self.t = self.input_batch[1] #tf.placeholder(tf.int32, [None])
        self.sign = self.input_batch[2] #tf.placeholder(tf.float32, [None])

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

    def train_one_epoch(self):
        sum_loss = 0.0
        batch_id = 0
        tot_time = 0.0
        start = time.time()
        numOfExamples = len(self.batches[0])

        # coord = tf.train.Coordinator()
        # numOfExamples = len(self.batches[0])
        # shuffle_indices = np.random.permutation(np.arange(len(self.batches[0])))
        # reader = DataGenerator(self.batches, shuffle_indices, self.negative_ratio, self.node_size, coord, max_queue_size=self.batch_size * 10)
        # self.input_batch = reader.dequeue(self.batch_size)
        threads = self.reader.start_threads(self.sess, len(self.batches[0]), n_threads=self.nthread)

        # h1, t1, sign = [],[],[]
        # for x in input_batch:
        #     h1.append(x[0])
        #     t1.append(x[1])
        #     sign.append(x[2])
        numOfBatches = int(numOfExamples / self.batch_size)
        print("Needs Total Batches: ", numOfBatches)
        for _ in range(numOfBatches):
            tx = time.time()
            # h1, t1, sign = self.sess.run(input_batch)
            _, cur_loss = self.sess.run([self.train_op, self.loss]) #, feed_dict={
                # self.h: h1, self.t: t1, self.sign: sign})
            tot_time += time.time() - tx
            sum_loss += cur_loss
            batch_id += 1
            if batch_id % 100 == 0:
                print("{}/{}, {!s}".format(batch_id, numOfBatches, tot_time/batch_id))

        print("Total Batches {}".format(batch_id))
        self.coord.request_stop()
        print("stop requested.")
        for thread in threads:
            thread.join()

        end = time.time()
        print('epoch {}: sum of loss:{!s}; time cost: {!s}/{!s}, per_batch_cost: {!s}'.
              format(self.cur_epoch, sum_loss / batch_id, tot_time, end - start, tot_time / batch_id))

    # def batch_iter(self):
    #     h0, t0 = self.batches
    #     shuffle_indices = np.random.permutation(np.arange(len(h0)))
    #
    #     table_size = len(shuffle_indices)
    #     i = 0
    #     while i < table_size:
    #         h = []
    #         t = []
    #         sign = []
    #         j = min(i + self.batch_size, table_size)
    #         # print(i,j)
    #         for k in range(i, j):
    #             idx = shuffle_indices[k]
    #             h.append(h0[idx])
    #             t.append(t0[idx])
    #             sign.append(1.0)
    #             for negId in range(self.negative_ratio):
    #                 h.append(h0[idx])
    #                 t.append(random.randint(0, self.node_size -1))
    #                 sign.append(0.0)
    #         yield h, t, sign
    #         i = j

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
        h, t = [], []
        with open(rw_file, 'r') as f:
            for line in f:
                walk = line.split()
                for i in range(len(walk)):
                    for j in range(i - self.window, i + self.window + 1):
                        if j > -1 and j != i and j < len(walk):
                            h.append(walk[i])
                            t.append(walk[j])
        return h, t


class DataGenerator(object):
    def __init__(self,
                 raw_data,
                 shuffled_index,
                 negative_ratio,
                 node_size,
                 coord,
                 max_queue_size=32,
                 wait_time=0.01):
        # Change the shape of the input data here with the parameter shapes.
        self.wait_time = wait_time
        self.max_queue_size = max_queue_size
        self.queue = tf.PaddingFIFOQueue(max_queue_size, dtypes=["int32", "int32", "float32"], shapes=[[],[],[]])
        self.queue_size = self.queue.size()
        self.threads = []
        self.coord = coord
        self.sample_h = tf.placeholder(dtype=tf.int32, shape=None)
        self.sample_t = tf.placeholder(dtype=tf.int32, shape=None)
        self.sample_sign = tf.placeholder(dtype=tf.float32, shape=None)
        self.enqueue = self.queue.enqueue([self.sample_h, self.sample_t, self.sample_sign])
        self.h, self.t = raw_data
        self.shuffled_index = shuffled_index
        self.negative_ratio = negative_ratio
        self.node_size = node_size

    def dequeue(self, num_elements):
        try:
            output = self.queue.dequeue_many(num_elements)
        except tf.errors.OutOfRangeError:
            return None
        return output

    def load_data(self, start, end):
        for k in range(start, end):
            idx = self.shuffled_index[k]
            yield self.h[idx], self.t[idx], 1.0
            for negId in range(self.negative_ratio):
                yield self.h[idx], random.randint(0, self.node_size-1), 0.0

    def thread_main(self, sess, start, end):

        iterator = self.load_data(start, end)
        cnt = 0;
        for data in iterator:
            while self.queue_size.eval(session=sess) == self.max_queue_size:
                if self.coord.should_stop():
                    break
                time.sleep(self.wait_time)
            if self.coord.should_stop():
                print("Enqueue thread receives stop request.")
                break
            cnt = cnt + 1
            # print("{},{}: {}".format(start, end, start+cnt))
            sess.run(self.enqueue, feed_dict={self.sample_h: data[0], self.sample_t: data[1], self.sample_sign: data[2]})
        print("Finish loading a part of data")

    def start_threads(self, sess, tot_size, n_threads=1):
        step = int((tot_size + n_threads - 1) / n_threads)
        for i in range(n_threads):
            start = i * step
            end = min((i + 1) * step, tot_size)
            thread = threading.Thread(target=self.thread_main, args=(sess, start, end))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
