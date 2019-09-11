# -*- coding: utf-8 -*-


import pickle
import shutil
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from model import TodAutoEncoder
from data_loader import get_batch
from utils import load_yaml_config, get_session, dense_transform


def train():
    config = load_yaml_config("config.yml")
    display_step = config["model"]["display_step"]
    evaluate_step = config["model"]["evaluate_step"]
    save_step = config["model"]["save_step"]
    checkpoint_path = config["model"]["checkpoint_path"]
    pickle_path = config["data"]["pickle_path"]
    pb_path = config["model"]["pb_path"]
    model = TodAutoEncoder(config)
    print(model.input_x)
    print(model.loss)

    with open(pickle_path, "rb") as f:
        _ = pickle.load(f)
        _, sparse_test = pickle.load(f)
    card, sparse = zip(*sparse_test)
    test = dense_transform(list(sparse))

    sess = get_session()
    sess.run(tf.global_variables_initializer())

    batch_data = get_batch()
    for batch in batch_data:
        _, loss_train, step = model.step(sess, batch)
        if step % display_step == 0:
            print("step: %d => loss: %.4f" % (step, loss_train))
        if step % evaluate_step == 0:
            _, loss_test, _ = model.step(sess, test)
            print("{0:-^30}".format("evaluation loss: %.4f" % loss_test))
            print("")
        if step % save_step == 0:
            model.save(sess, checkpoint_path)
    model.save(sess, checkpoint_path)

    shutil.rmtree(pb_path, ignore_errors=True)
    builder = tf.saved_model.builder.SavedModelBuilder(pb_path)
    inputs = {'input_x': tf.saved_model.utils.build_tensor_info(model.input_x)}
    outputs = {'output': tf.saved_model.utils.build_tensor_info(model.loss)}
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'my_signature': signature})
    builder.save()


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
