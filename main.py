import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo.yolo_v3
import yolo.yolo_v3_tiny
import util.config as config
from util import util


def main(argv=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.GPU_MEMORY_FRACTION)

    conf = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    img = Image.open(config.INPUT_IMAGE)
    img_resized = util.letter_box_image(img, config.MODEL_SIZE, config.MODEL_SIZE, 128)
    img_resized = img_resized.astype(np.float32)
    classes = util.load_class_names(config.CLASSES_FILE)

    if config.TINY:
        model = yolo.yolo_v3_tiny.yolo_v3_tiny
    elif config.SPP:
        model = yolo.yolo_v3.yolo_v3_spp
    else:
        model = yolo.yolo_v3.yolo_v3

    boxes, inputs = util.get_boxes_and_inputs(model, len(classes), config.MODEL_SIZE, config.DATA_FORMAT)

    saver = tf.train.Saver(var_list=tf.global_variables(scope="detector"))

    with tf.Session(config=conf) as sess:
        t0 = time.time()
        saver.restore(sess, config.CHECKPOINT_FILE)
        print("Model restored in {:.2f}s".format(time.time() - t0))

        t0 = time.time()
        detected_boxes = sess.run(boxes, feed_dict={inputs: [img_resized]})

    filtered_boxes = util.non_max_suppression(detected_boxes,
                                              confidence_threshold=config.CONF_THRESHOLD,
                                              iou_threshold=config.IOU_THRESHOLD)

    print("Predictions found in {:.2f}s".format(time.time() - t0))

    util.draw_boxes(filtered_boxes, img, classes, (config.MODEL_SIZE, config.MODEL_SIZE), True)

    print(img.filename)

    img.save(config.OUTPUT_DIR + img.filename.split("/")[2])


if __name__ == "__main__":
    tf.app.run()
