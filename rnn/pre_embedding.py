# -*- coding: utf-8 -*-


import logging
import multiprocessing

import jieba
from gensim.models import word2vec

from utils import load_yaml_config


def train():
    config = load_yaml_config("config.yml")
    embedding_size = config["model"]["embedding_size"]
    min_count = config["model"]["min_count"]
    sg = config["model"]["sg"]
    output_path = config["data"]["output_path"]
    window = config["model"]["window"]
    embedding_model = config["model"]["embedding_model"]
    embedding_path = config["model"]["embedding_path"]

    sentences = word2vec.LineSentence(output_path)
    model = word2vec.Word2Vec(sentences, size=embedding_size, window=window,
                              min_count=min_count, sg=sg,
                              workers=multiprocessing.cpu_count())
    print("word counts: %d" % len(model.wv.vocab.keys()))
    model.save(embedding_model)
    model.wv.save_word2vec_format(embedding_path, binary=False)


if __name__ == "__main__":
    jieba.setLogLevel(logging.INFO)
    train()
