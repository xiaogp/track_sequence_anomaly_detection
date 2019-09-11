# -*- coding: utf-8 -*-


import random
import pickle
from utils import load_yaml_config, dense_transform


def load_train_pkl(pickle_path):
    with open(pickle_path, "rb") as f:
        _ = pickle.load(f)
        sparse_train, _ = pickle.load(f)

    return sparse_train


def get_batch():
    config = load_yaml_config("config.yml")
    pickle_path = config["data"]["pickle_path"]
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]

    sparse_list = load_train_pkl(pickle_path)
    card, sparse = zip(*sparse_list)
    sparse = list(sparse)
    data_length = len(sparse)

    # repeat
    for epoch in range(epochs):
        # shuffle
        random.shuffle(sparse)
        # batch_size
        for batch in range(0, data_length, batch_size):
            if batch + batch_size <= data_length:
                sparse_batch = sparse[batch: (batch + batch_size)]
                # 对稀疏矩阵进行dense转化
                dense_list = dense_transform(sparse_batch)

                yield dense_list


if __name__ == "__main__":
    batch_data = get_batch()

    for num, data in enumerate(batch_data):
        print("{0:-^30}".format("第%d个batch" % (num + 1)))
        print("batch样本数量:", len(data))
        print("一条样本的维度:", len(data[0]))
        print("出现的时空点数量:", sum(data[0]))
        if num == 3:
            break