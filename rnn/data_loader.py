# -*- coding: utf-8 -*-


import random
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import load_yaml_config

config = load_yaml_config("config.yml")
output_path = config["data"]["output_path"]
tokenizer_path = config["data"]["tokenizer_path"]
seq_length = config["model"]["seq_length"]
test_path = config["data"]["test_path"]
epochs = config["model"]["epochs"]
batch_size = config["model"]["batch_size"]
train_path = config["data"]["train_path"]


def tokenizer_seq():
    with open(output_path, "r", encoding="utf8") as f:
        track = [line.strip() for line in f.readlines()]

    # 使用分词器，保留所有的词，不过滤标点符号
    tokenizer = tf.contrib.keras.preprocessing.text.Tokenizer(split=" ", oov_token="<unk>", filters="")
    tokenizer.fit_on_texts(track)
    pickle.dump(tokenizer, open(tokenizer_path, "wb"))

    print("{0:-^30}".format("词频统计top10"))
    for st, count in sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(st, count)

    # 对所有行进行字符串转索引
    tokenizerSeq = tokenizer.texts_to_sequences(track)

    # 以序列长度作为长度构造序列
    featureList = []
    labelList = []
    for line in tokenizerSeq:
        if len(line) <= seq_length:
            if len(line) == 1:
                continue
            label = line[-1]
            feature = [1] * (seq_length + 1 - len(line)) + line[:-1]
            featureList.append(feature)
            labelList.append(label)

        else:
            for i in range(len(line) - seq_length):
                feature = line[i: i + seq_length]
                label = line[i + seq_length]
                featureList.append(feature)
                labelList.append(label)

    # train_test_split
    train_x, test_x, train_y, test_y = train_test_split(featureList, labelList, test_size=0.1)
    pickle.dump((train_x, train_y), open(train_path, "wb"))
    pickle.dump((test_x, test_y), open(test_path, "wb"))


def get_batch():
    train_data = pickle.load(open(train_path, "rb"))
    data = list(zip(train_data[0], train_data[1]))
    for epoch in range(epochs):
        random.shuffle(data)  # shuffle
        for batch in range(0, len(data), batch_size):
            if batch + batch_size <= len(data):
                yield data[batch: (batch + batch_size)]


if __name__ == "__main__":
    tokenizer_seq()
    batch_data = get_batch()
    for num, batch in enumerate(batch_data):
        print("{0:-^30}".format("batch %d" % (num + 1)))
        print("batch num:", len(batch))
        feature, label = zip(*batch)
        for i in range(len(batch)):
            print(feature[i], label[i])
        break
