# -*- coding: utf-8 -*-


import numpy as np
import pickle
from scipy import stats
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from utils import load_yaml_config


def main():
    config = load_yaml_config("config.yml")
    tokenizer_path = config["data"]["tokenizer_path"]
    seq_length = config["model"]["seq_length"]
    pb_path = config["model"]["pb_path"]
    label_size = config["model"]["label_size"]
    kl_threshold = config["model"]["kl_threshold"]

    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], pb_path)
        graph = tf.get_default_graph()

        input_x = graph.get_tensor_by_name("input_x:0")
        keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        pred = graph.get_tensor_by_name("softmaxLayer/probs:0")

        while True:
            trackString = input("请输入轨迹序列:")
            if trackString == "exit":
                print("退出检测")
                exit(0)

            track = trackString.strip().split(" ")
            if len(track) < 2:
                print("轨迹长度至少为2")
                continue

            unknow = [x for x in track if x not in tokenizer.word_index]
            if unknow:
                print("{}轨迹点不存在".format(unknow))
                continue

            tokenizerSeq = tokenizer.texts_to_sequences([trackString])[0]
            for i in range(1, len(tokenizerSeq)):
                label = tokenizerSeq[i]
                feature = tokenizerSeq[(i - 4) if (i - 4) >= 0 else 0:i]
                if len(feature) < seq_length:
                    feature = [1] * (seq_length - len(feature)) + feature
                feature_reshape = np.array(feature).reshape(-1, 4)
                logits = sess.run(pred, feed_dict={
                    input_x: feature_reshape,
                    keep_prob: 1})
                # 多分类交叉熵
                loss = sess.run(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=[label]))[0]
                # kl离散度
                onehot = [1e-6] * label_size
                onehot[label] = 1
                kl = stats.entropy(logits[0], onehot)
                # 用kl离散度判断异常
                abnormal = "[{}]".format(" " if kl <= kl_threshold else "×")
                print("{} 交叉熵：{:.4f}, kl离散度：{:.4f}, 移动轨迹：{} => {}".format(
                    abnormal, loss, kl, track[(i - 4) if (i - 4) >= 0 else 0:i], track[i]))
            print("")


if __name__ == "__main__":
    main()
