# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size,
                 rnn_size, label_size, embedding=None, use_bilstm=False):

        self.input_x = tf.placeholder(tf.int64, [None, seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, 1])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope('embeddingLayer'):
            if embedding:
                embedded = tf.nn.embedding_lookup(embedding, self.input_x)
            else:
                W = tf.get_variable('W', [vocab_size, embedding_size])
                embedded = tf.nn.embedding_lookup(W, self.input_x)

            inputs = tf.split(embedded, seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def get_cell(rnn_size, dropout_keep_prob):
            cell = rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.3))
            cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)  # dropout比例
            return cell

        with tf.name_scope('lstm_layer'):
            if use_bilstm:
                cell_fw = rnn.MultiRNNCell([get_cell(rnn_size, self.dropout_keep_prob) for _ in range(num_layers)])
                cell_bw = rnn.MultiRNNCell([get_cell(rnn_size, self.dropout_keep_prob) for _ in
                                            range(num_layers)])
                self.outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw=cell_fw,
                                                                  cell_bw=cell_bw,
                                                                  inputs=inputs,
                                                                  dtype=tf.float32)
            else:
                cell = rnn.MultiRNNCell([get_cell(rnn_size, self.dropout_keep_prob) for _ in range(num_layers)])
                self.outputs, _ = rnn.static_rnn(cell, inputs, dtype=tf.float32)

        with tf.name_scope('softmaxLayer'):
            logits = tf.layers.dense(self.outputs[-1], label_size)
            self.probs = tf.nn.softmax(logits, dim=1, name="probs")

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(
                self.input_y)))
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)
