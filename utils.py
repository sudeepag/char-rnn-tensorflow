import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import pandas as pd
import re

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.num_batches = 0

        # input_file = os.path.join(data_dir, "input.txt")
        input_file = os.path.join(data_dir, "input.csv")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.x_batches = []
        self.y_batches = []
        for tensor in self.tensors:
            x_batches, y_batches = self.create_batches(tensor)
            if x_batches and y_batches:
                self.x_batches += x_batches
                self.y_batches += y_batches
        # print('self.x_batches[0]:', self.x_batches[0].shape)
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        poems = list(pd.read_csv(input_file, header=None)[0])
        regex = re.compile('[^a-zA-Z]')
        poems = [' '.join([e.strip() for e in regex.sub(' ', p.replace("'", '').replace('’', '')).lower().split(' ') if len(e.strip()) > 0]) for p in poems]
        data = ' '.join(poems)
        # with codecs.open(input_file, "r", encoding=self.encoding) as f:
        #     data = f.read()
        print('data:', data)
        print('datalen', len(data))
        counter = collections.Counter(data)
        # print('counter:', counter)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # print('count_pairs:', count_pairs)
        self.chars, _ = zip(*count_pairs)
        # print('chars:', self.chars)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        # print('vocab:', self.vocab)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensors = [np.array(list(map(self.vocab.get, p))) for p in poems]
        # print('tensor:', self.tensor)
        # print('shape', self.tensor.shape)
        np.save(tensor_file, self.tensors)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self, tensor):
        num_batches = int(tensor.size / (self.batch_size *
                                                   self.seq_length))
        self.num_batches += num_batches
        # When the data (tensor) is too small,
        # let's give them a better error message
        if num_batches == 0:
            return None, None
        #     assert False, "Not enough data. Make seq_length and batch_size small."

        tensor = tensor[:num_batches * self.batch_size * self.seq_length]
        xdata = tensor
        ydata = np.copy(tensor)
        print('old ydata:', ydata)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        print('new ydata:', ydata)
        x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  num_batches, 1)
        y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  num_batches, 1)
        # print('x batch:', self.x_batches[0])
        # print('y batch:', self.y_batches[0])
        return x_batches, y_batches

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
