import tensorflow as tf

import yolo.yolo_v3
import yolo.yolo_v3_tiny
import util.config as config
from util import util


def main(argv=None):
    classes = util.load_class_names(config.CLASSES_FILE)

    if config.TINY:
        model = yolo.yolo_v3_tiny.yolo_v3_tiny
    elif config.SPP:
        model = yolo.yolo_v3.yolo_v3_spp
    else:
        model = yolo.yolo_v3.yolo_v3

        # placeholder for detector inputs
        # any size > 320 will work here
        inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])

        with tf.variable_scope("detector"):
            detections = model(inputs, len(classes),
                               data_format=config.DATA_FORMAT)
            load_ops = util.load_weights(tf.global_variables(scope="detector"),
                                         config.WEIGHTS_FILE)

        saver = tf.train.Saver(tf.global_variables(scope="detector"))

        with tf.Session() as sess:
            sess.run(load_ops)

            save_path = saver.save(sess, save_path=config.CHECKPOINT_FILE)
            print("Model saved in path: {}".format(save_path))


if __name__ == "__main__":
    tf.app.run()
