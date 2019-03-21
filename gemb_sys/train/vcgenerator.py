from __future__ import print_function
import random


class VCGenerator(object):

    def __init__(self):
        random.seed()

    def generate_batch(self, batch_size):
        raise NotImplementedError

