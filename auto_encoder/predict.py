# -*- coding: utf-8 -*-


import pickle
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from utils import load_yaml_config


def main():
    config = load_yaml_config("config.yml")
    pickle_path = config["data"]["pickle_path"]
    pb_path = config["model"]["pb_path"]

    with open(pickle_path, "rb") as f:
        _ = pickle.load(f)
        sparse_train, sparse_test = pickle.load(f)
    card_st = sparse_train + sparse_test

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], pb_path)
        graph = tf.get_default_graph()

        input_x = graph.get_tensor_by_name("input_x:0")
        pred = graph.get_tensor_by_name("cost/absolute_difference/value:0")
        # 预测
        with open("tod_result.csv", "w", encoding="utf8") as f:
            f.write("{},{}\n".format("cardno", "loss"))

            for card, st_sparse in card_st:
                st_dense = st_sparse.toarray().reshape(1, -1)
                res = sess.run(pred, feed_dict={input_x: st_dense})
                f.write("{},{}\n".format(card, res))


if __name__ == "__main__":
    main()
