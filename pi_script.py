import numpy as np
import tensorflow as tf
from PIL import Image
import time
import glob
import cv2

import yolo.yolo_v3
import yolo.yolo_v3_tiny
import util.config as config
from util import util
import camera_capture as cam

# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db


def main(argv=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.GPU_MEMORY_FRACTION)

    conf = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    # image_paths = glob.glob(config.IMAGES_DIR + "*.jpg")
    # input_images = util.load_images(image_paths, config.MODEL_SIZE)

    # get capture
    image_path = cam.get_capture()

    # make detections
    input_image = util.load_images(image_path, config.MODEL_SIZE)

    classes = util.load_class_names(config.CLASSES_FILE)

    if config.TINY:
        model = yolo.yolo_v3_tiny.yolo_v3_tiny
    elif config.SPP:
        model = yolo.yolo_v3.yolo_v3_spp
    else:
        model = yolo.yolo_v3.yolo_v3

    # TODO : ugly
    boxes, inputs = util.get_boxes_and_inputs(model, len(input_image),
                                              len(classes), config.MODEL_SIZE, config.DATA_FORMAT)

    saver = tf.train.Saver(var_list=tf.global_variables(scope="detector"))
    input_dict = {inputs: input_image}
    with tf.Session(config=conf) as sess:
        t0 = time.time()
        saver.restore(sess, config.CHECKPOINT_FILE)
        print("Model restored in {:.2f}s".format(time.time() - t0))

        t0 = time.time()

        detected_boxes = sess.run(boxes, feed_dict=input_dict)
        print("Predictions found in {:.2f}s".format(time.time() - t0))

    filtered_boxes = util.non_max_suppression(detected_boxes,
                                              confidence_threshold=config.CONF_THRESHOLD,
                                              iou_threshold=config.IOU_THRESHOLD)

    # print(filtered_boxes)
    print("~~~~~~~~~~~~~~~~~~~~~~~~")
    # for key in filtered_boxes:
    #     print(key, " : ", filtered_boxes[key])

    print(input_dict)

    print(len(filtered_boxes))

    # for i in range(len(input_images)):
    #     box = filtered_boxes[i]
    #     image_name = image_paths[i].split("/")[2]
    #     image = input_images[i]
    #
    #     util.draw_boxes(box, image, classes, (config.MODEL_SIZE, config.MODEL_SIZE), True)
    #
    #     image.save(config.OUTPUT_DIR + image_name)
    # #
    # # print(img.filename)
    # #

    # # Firebase connection
    # TODO : add a config variable for the certificate location
    # cred = credentials.Certificate(r"/Users/angie/Downloads/Firebase_Connection/smartmirrai-c2051-firebase-adminsdk"
    #                                r"-pnq5k-19437db28a.json")
    # # Access Firebase DB
    # firebase_admin.initialize_app(cred, {'databaseURL': 'https://smartmirrai-c2051-default-rtdb.firebaseio.com/'})

    # Fetch and print firebase DB
    # ref = db.reference('images')
    # print(ref.get())

    # TODO
    # send detections to firebase

    # TODO
    # wait to receive 4 images from firebase

    # TODO
    # save images from firebase to "recos" directory


if __name__ == "__main__":
    tf.app.run()
