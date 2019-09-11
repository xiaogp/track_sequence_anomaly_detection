# -*- coding: utf-8 -*-


import os
import pickle
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from data_loader import get_batch
from model import Model
from utils import load_yaml_config

config = load_yaml_config("config.yml")
num_layers = config["model"]["num_layers"]
seq_length = config["model"]["seq_length"]
embedding_size = config["model"]["embedding_size"]
vocab_size = config["model"]["vocab_size"]
rnn_size = config["model"]["rnn_size"]
use_bilstm = config["model"]["use_bilstm"]
learning_rate = config["model"]["learning_rate"]
summary_path = config["model"]["summary_path"]
test_path = config["data"]["test_path"]
cktp_path = config["model"]["cktp_path"]
pb_path = config["model"]["pb_path"]
label_size = config["model"]["label_size"]
use_word2vec = config["model"]["use_word2vec"]
fix_embedding = config["model"]["fix_embedding"]
tokenizer_path = config["data"]["tokenizer_path"]
embedding_path = config["model"]["embedding_path"]


def load_word_embedding(vocab_size, token2index):
    embedding_np = 0.3 * \
                   np.random.randn(vocab_size, embedding_size).astype("float32")

    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.replace("\n", "").strip().split()
            token = tokens[0]
            vector = tokens[1:]
            if token in token2index:
                embedding_np[token2index[token], :] = vector

    embedding = tf.get_variable("embedding",
                                shape=[vocab_size, embedding_size],
                                initializer=tf.constant_initializer(embedding_np),
                                trainable=fix_embedding)
    return embedding


def train():
    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    token2index = tokenizer.word_index

    if use_word2vec:
        embedding = load_word_embedding(vocab_size, token2index)

    lstm = Model(num_layers=num_layers,
                 seq_length=seq_length,
                 embedding_size=embedding_size,
                 vocab_size=vocab_size,
                 rnn_size=rnn_size,
                 use_bilstm=use_bilstm,
                 label_size=label_size,
                 embedding=embedding if use_word2vec else None)

    print("{0:-^40}".format("需要训练的参数"))
    for var in tf.trainable_variables():
        print(var.name, var.shape)

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lstm.loss, global_step=global_step)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        shutil.rmtree(summary_path, ignore_errors=True)
        writer = tf.summary.FileWriter(summary_path, sess.graph)

        def train_step(batch, label):
            feed_dict = {
                lstm.input_x: batch,
                lstm.input_y: label,
                lstm.dropout_keep_prob: 0.7
            }
            _, step, loss, accuracy = sess.run([optimizer, global_step, lstm.loss, lstm.accuracy], feed_dict=feed_dict)

            return step, loss, accuracy

        def dev_step(batch, label):
            feed_dict = {
                lstm.input_x: batch,
                lstm.input_y: label,
                lstm.dropout_keep_prob: 1
            }
            step, loss, accuracy = sess.run([global_step, lstm.loss, lstm.accuracy], feed_dict=feed_dict)
            print("{0:-^40}".format("evaluate"))
            print("step:{}".format(step), "==>", "loss:%.5f" % loss, "accuracy:%.5f" % accuracy)

        batches = get_batch()
        x_dev, y_dev = pickle.load(open(test_path, "rb"))
        y_dev = np.array(y_dev).reshape(-1, 1)

        print("{0:-^40}".format("模型训练"))
        for data in batches:
            x_train, y_train = zip(*data)
            y_train = np.array(y_train).reshape(-1, 1)
            step, loss, accuracy = train_step(x_train, y_train)
            current_step = tf.train.global_step(sess, global_step)
            result = sess.run(merged, feed_dict={
                lstm.input_x: x_train,
                lstm.input_y: y_train,
                lstm.dropout_keep_prob: 0.5})
            writer.add_summary(result, current_step)

            if current_step % 10 == 0:
                print("global step:{}".format(step), "==>", "loss:%.5f" % loss, "accuracy:%.5f" % accuracy)

            if current_step % 500 == 0:
                dev_step(x_dev, y_dev)
                print("")
        dev_step(x_dev, y_dev)

        saver.save(sess, cktp_path, global_step=current_step)

        shutil.rmtree(pb_path, ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder(pb_path)
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(lstm.input_x),
                  'dropout_keep_prob': tf.saved_model.utils.build_tensor_info(lstm.dropout_keep_prob)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(lstm.probs)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'my_signature': signature})
        builder.save()


def main(argv=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
