# -*- coding: utf-8 -*-


import yaml
import tensorflow as tf


def get_session():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    return tf.Session(config=sess_config)


def load_yaml_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f)
    return config


def dense_transform(sparse_list):
    return [x.toarray().reshape(1, -1)[0].tolist() for x in sparse_list]
