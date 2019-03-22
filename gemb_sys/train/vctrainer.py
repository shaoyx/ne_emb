from __future__ import print_function
import random
import tensorflow as tf
import time


class vctrainer(object):
    def __init__(self,
                 graph,
                 vsampler, csampler, emb_model, emb_file,
                 rep_size=128, epoch=10, batch_size=1000, learning_rate=0.001, negative_ratio=5):
        self.g = graph
        self.model_v = vsampler
        self.model_c = csampler
        self.emb_model = emb_model #sym (first), asym (second)
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.emb_file = emb_file
        self.cur_epoch = 0

        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=cur_seed)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_model(self.emb_model)

        self.sess.run(tf.global_variables_initializer())
        print("Start training.")

        for i in range(epoch):
            self.train_one_epoch()
            self.cur_epoch += 1
            if self.cur_epoch % 5 == 0:
                self.get_embeddings()
                self.save_embeddings(self.emb_file+"_"+str(self.cur_epoch))
        self.get_embeddings()

    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.eval(session=self.sess)
        # embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        self.vectors = vectors

    def build_model(self, emb_model):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])
        # self.step = tf.placeholder(tf.float32, shape=())

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
        # lr = max(0.0001, 100/pow(10, self.cur_epoch-1))
        # lr = tf.maximum(0.0001, 100/pow(10.0, self.step-1))
        # print(lr)
        # lr = 1000
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #need large learning rate and lr decay schedule
        # self.grads_and_vars = optimizer.compute_gradients(self.loss)
        # self.train_op = optimizer.apply_gradients(self.grads_and_vars)
        self.train_op = optimizer.minimize(self.loss)

    """
       positive and negative batches seperately.
    """
    def train_one_epoch(self):
        sum_loss = 0.0
        batch_id = 0
        tot_time = 0.0
        start = time.time()
        for batch in self.model_v.generate_batch(self.batch_size):
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

            # tx = time.time()
            # _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
                                # self.h: h1, self.t: t1, self.sign: [1.0]})
            # _, cur_loss, grads = self.sess.run([self.train_op, self.loss, self.grads_and_vars], feed_dict={
            #     self.h: h1, self.t: t1, self.sign: [1.0]})
            # for (g,v) in grads:
            #     print(g)
            # print("################")
            # tot_time += time.time() - tx
            # sum_loss += cur_loss
            # for i in range(self.negative_ratio):
            #     t1 = self.neg_batch(h1)
            #     tx = time.time()
            #     _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
            #                     self.h: h1, self.t: t1, self.sign: [0.0]})
            #     tot_time += time.time() - tx
            #     sum_loss += cur_loss
            # for gv in self.grads_and_vars:
            #     print(str(self.sess.run(gv[0],  feed_dict={
            #                     self.h: h1, self.t: t1, self.sign: [0.0]}))+"-"+gv[1].name)

            # h1, t1 = batch
            # tx = time.time()
            # _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
            #                     self.h: h1, self.t: t1, self.sign: [1.0]})
            # tot_time += time.time() - tx
            # sum_loss += cur_loss
            # for i in range(self.negative_ratio):
            #     t1 = self.neg_batch(h1)
            #     tx = time.time()
            #     _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
            #                     self.h: h1, self.t: t1, self.sign: [0.0]})
            #     tot_time += time.time() - tx
            #     sum_loss += cur_loss
            # batch_id += 1
        end = time.time()

        print('epoch {}: sum of loss:{!s}; time cost: {!s}/{!s}, per_batch_cost: {!s}'.
              format(self.cur_epoch, sum_loss / batch_id, tot_time, end-start, tot_time/batch_id))

    """
    mix negative batch and positive batches
    """
    # def train_one_epoch(self):
    #     sum_loss = 0.0
    #
    #     batch_id = 0
    #
    #     tot_time = 0.0
    #     start = time.time()
    #     for batch in self.model_v.sample_batch(self.batch_size, self.negative_ratio):
    #         h1, t1, sign = batch
    #         tx = time.time()
    #         _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
    #                             self.h: h1, self.t: t1, self.sign: sign})
    #         tot_time += time.time() - tx
    #         sum_loss += cur_loss
    #         batch_id += 1
    #     end = time.time()
    #
    #     print('epoch {}: sum of loss:{!s}; time cost: {!s}/{!s}, per_batch_cost: {!s}'.
    #           format(self.cur_epoch, sum_loss / batch_id, tot_time, end-start, tot_time/batch_id))

    # def train_one_epoch1(self):
    #     sum_loss = 0.0
    #     vs = self.model_v.sample_v(self.batch_size)
    #     batch_id = 0
    #     for hx in vs:
    #         # TODO, return two lists
    #         h1, t1 = self.model_c.sample_c(hx)
    #         sign = [1.]
    #         _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
    #                             self.h: h1, self.t: t1, self.sign: sign})
    #         sum_loss += cur_loss
    #         batch_id += 1 #positive batch
    #         for i in range(self.negative_ratio):
    #             t1 = self.neg_batch(h1)
    #             sign = [-1.]
    #             _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
    #                             self.h: h1, self.t: t1, self.sign: sign})
    #             sum_loss += cur_loss
    #             # print('\tBatch {}: loss:{!s}/{!s}'.format(batch_id, cur_loss, sum_loss))
    #             batch_id += 1
    #
    #     print('epoch {}: sum of loss:{!s}'.format(self.cur_epoch, sum_loss / batch_id))

    def neg_batch(self, h):
        t = []
        for i in range(len(h)):
            t.append(random.randint(0, self.node_size-1))
        return t

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

