
import tensorflow as tf

import random
import string

from dnc import DNC

from tensorflow.contrib import legacy_seq2seq


def loss_function(predictions, targets):
    return tf.reduce_mean(
        -1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions)
    )


def random_string(length):
    return ''.join([random.choice(string.ascii_lowercase) for _ in range(length)])


def get_train_data(batch_size, x):
    for i in range(x):
        r = []
        for n in range(batch_size):
            x = random_string(10)
            y = x.title()
            r.append((x, y))
        yield r


def char_to_vec(ch):
    vec = [0] * 256
    vec[ord(ch)] = 1
    return vec


def output_to_string(m):
    r = []
    for i in range(8):
        s = ''
        for v in m:
            v = v[i]
            max_index = 0
            max_value = v[0]
            for idx, value in enumerate(v):
                if value > max_value:
                    max_value = value
                    max_index = idx
            s += chr(max_index)
        r.append(s)
    return r


access_config = {
    "memory_size": 16,
    "word_size": 16,
    "num_reads": 4,
    "num_writes": 1,
}

controller_config = {
  "hidden_size": 64,
}

clip_value = 20
data_point_dim = 256


class Model(object):

    def __init__(self, batch_size, input_seq_length, output_seq_length):

        self._encoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim], name='inputs_{}'.format(i)) for i in xrange(input_seq_length)]
        self._labels = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim], name='labels_{}'.format(i)) for i in xrange(output_seq_length)]
        self._decoder_inputs = [tf.zeros_like(self._encoder_inputs[0], dtype=tf.float32, name='GO')] + self._labels[:-1]
        rnn_cell = DNC(access_config, controller_config, data_point_dim, clip_value)

        model_outputs, states = legacy_seq2seq.tied_rnn_seq2seq(self._encoder_inputs,
                                                                self._decoder_inputs,
                                                                rnn_cell,
                                                                loop_function=lambda prev, _: prev)
        self._batch_size = batch_size
        self._input_seq_length = input_seq_length
        self._output_seq_length = output_seq_length
        self._squashed_output = tf.nn.softmax(model_outputs)
        self._cost = loss_function(tf.reshape(self._squashed_output, [-1]), tf.reshape(self._labels, [-1]))
        self._step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self._cost)
        self._session = tf.Session()

        init = tf.global_variables_initializer()
        self._session.run(init)

    def _partition_by_batch_size(self, data):
        for i in range(0, len(data), self._batch_size):
            yield data[i:i + self._batch_size]

    def fit(self, X, Y, epoch):

        if len(X) != len(Y):
            raise ValueError('x, y must have same length')

        i = 0
        for n in range(epoch):
            for batch in self._partition_by_batch_size(zip(X, Y)):
                feed_dict = {}
                for i in range(self._input_seq_length):
                    feed_dict[self._encoder_inputs[i]] = []
                for i in range(self._output_seq_length):
                    feed_dict[self._labels[i]] = []
                for x, y in batch:
                    for i in range(self._input_seq_length):
                        feed_dict[self._encoder_inputs[i]].append(char_to_vec(x[i]))
                    for i in range(self._output_seq_length):
                        feed_dict[self._labels[i]].append(char_to_vec(y[i]))
                c, _ = self._session.run([self._cost, self._step], feed_dict=feed_dict)
                i += 1
                if i % 100 == 0:
                    print 'epoch: {}, Cost: {}'.format(n, c)

    def predict(self, X):
        for batch in self._partition_by_batch_size(X):
            if len(batch) < self._batch_size:
                batch = batch + [' ' * self._input_seq_length] * (self._batch_size - len(batch))
            feed_dict = {}
            for i in range(self._input_seq_length):
                feed_dict[self._encoder_inputs[i]] = []
            for x in batch:
                print x
                for i in range(self._input_seq_length):
                    feed_dict[self._encoder_inputs[i]].append(char_to_vec(x[i]))
            output, = self._session.run([self._squashed_output], feed_dict=feed_dict)
            print output_to_string(output)


if __name__ == '__main__':

    X, Y = [], []

    for n in range(5000):
        x = random_string(10)
        X.append(x)
        Y.append(x.title())

    model = Model(8, 10, 10)
    model.fit(X, Y, 10)

    X = []
    for n in range(10):
        x = random_string(10)
        X.append(x)

    model.predict(X)
