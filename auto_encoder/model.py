# -*- coding: utf-8 -*-


import os

import tensorflow as tf


class TodAutoEncoder(object):
    def __init__(self, config):
        self.learning_rate = config["model"]["learning_rate"]
        self.n_hidden_1 = config["model"]["n_hidden_1"]
        self.n_hidden_2 = config["model"]["n_hidden_2"]
        self.n_input = config["model"]["n_input"]

        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input], name="input_x")
        self.label = self.input_x
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.name_scope("encoder"):
            encoder_layer1 = tf.layers.dense(self.input_x, self.n_hidden_1, activation=tf.nn.sigmoid)
            encoder_layer2 = tf.layers.dense(encoder_layer1, self.n_hidden_2, activation=tf.nn.sigmoid)

        with tf.name_scope("decoder"):
            decoder_layer1 = tf.layers.dense(encoder_layer2, self.n_hidden_1, activation=tf.nn.sigmoid)
            decoder_layer2 = tf.layers.dense(decoder_layer1, self.n_input, activation=tf.nn.sigmoid)

        with tf.name_scope("cost"):
            self.loss = tf.losses.absolute_difference(labels=self.label, predictions=decoder_layer2)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss, global_step=self.global_step)

    def step(self, sess, one_batch):
        output_feeds = [self.train_op, self.loss, self.global_step]
        feed_dict = {self.input_x: one_batch}

        result = sess.run(output_feeds, feed_dict=feed_dict)
        return result

    def save(self, session, checkpoint_path, checkpoint_name="model.cktp"):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        print('[*] saving checkpoints to {}...'.format(checkpoint_path))
        self.saver.save(session, os.path.join(checkpoint_path, checkpoint_name))
