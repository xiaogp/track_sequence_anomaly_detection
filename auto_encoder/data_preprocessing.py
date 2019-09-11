# -*- coding: utf-8 -*-


import time
from scipy.sparse import dok_matrix
import math
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from utils import load_yaml_config


def get_space_time_dict():
    cstCountDict = {}
    cardCountDict = {}
    stCountDict = {}
    locCountDict = {}
    hourCountDict = {}

    with open(file_path, "r", encoding="utf8") as f:
        next(f)
        for line in f.readlines():
            loc, tim = [int(x) for x in line.strip().split(",")[1:3]]
            card = line.strip().split(",")[6]
            date_hour = time.localtime(float(tim / 1000)).tm_hour

            # {card:{loc:{hour: count}}}
            cstCountDict.setdefault(card, {})
            cstCountDict[card].setdefault(loc, {})
            cstCountDict[card][loc][date_hour] = cstCountDict[card][loc].get(date_hour, 0) + 1

            # {card: count}
            cardCountDict[card] = cardCountDict.get(card, 0) + 1

            # {loc: {hour: set(card)}}
            stCountDict.setdefault(loc, {})
            stCountDict[loc].setdefault(date_hour, set())
            stCountDict[loc][date_hour].add(card)
            # {loc: {hour: card_num}
            stCardDict = {k: {kk: len(vv) for kk, vv in v.items()} for k, v in stCountDict.items()}

            # {loc: Count}
            locCountDict[loc] = locCountDict.get(loc, 0) + 1

            # {hour: Count}
            hourCountDict[date_hour] = hourCountDict.get(date_hour, 0) + 1

    # 计算tf-idf
    num = len(hourCountDict) * len(locCountDict)
    card_sum = len(cardCountDict)
    k = 1

    tfidfCstDict = {}
    for card, stDict in cstCountDict.items():
        for space, hourDict in stDict.items():
            for hour, count in hourDict.items():
                tf = (count + k) / (cardCountDict[card] + num * k)  # 平滑处理
                tfidf = tf * math.log(card_sum / stCardDict[space][hour])
                tfidfCstDict.setdefault(card, {})
                tfidfCstDict[card].setdefault(space, {})
                tfidfCstDict[card][space][hour] = tfidf

    return tfidfCstDict, locCountDict, hourCountDict


def get_space_time_sparse(tfidfCstDict, locCountDict, hourCountDict):
    loc_length, hour_length = len(locCountDict), len(hourCountDict)
    # 对loc做索引变换
    locToIndex = {loc: index for index, loc in enumerate(locCountDict)}

    sparse_list = []
    for card, st_dict in tfidfCstDict.items():
        # 创建稀疏矩阵
        sparse_mat = dok_matrix((loc_length, hour_length), dtype=np.float32)
        for space, hour_dict in st_dict.items():
            for hour, tfidf in hour_dict.items():
                space_index = locToIndex[space]
                sparse_mat[space_index, hour] = tfidf
        sparse_list.append((card, sparse_mat))

    return sparse_list, locToIndex


def dump_pickel():
    tfidfCstDict, locCountDict, hourCountDict = get_space_time_dict()
    sparse_list, locToIndex = get_space_time_sparse(tfidfCstDict, locCountDict, hourCountDict)
    # train_test_split
    sparse_train, sparse_test = train_test_split(sparse_list, test_size=0.2)

    with open(pickle_path, "wb") as f:
        pickle.dump(locToIndex, f)
        pickle.dump((sparse_train, sparse_test), f, protocol=0)


if __name__ == "__main__":
    config = load_yaml_config("config.yml")
    file_path = config["data"]["file_path"]
    pickle_path = config["data"]["pickle_path"]
    dump_pickel()
