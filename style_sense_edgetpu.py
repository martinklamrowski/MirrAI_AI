import time
import cv2
import os

import picamera
from picamera.array import PiRGBArray

from pycoral.adapters.common import set_resized_input
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

import numpy as np

import config.config as cfg


def main():
    model_path = "../stylesense/data/mobilenetv2_stylesense16_quant_int8_edgetpu.tflite"
    labels_path = "../stylesense/data/stylesense_labels.txt"

    with picamera.PiCamera() as camera:

        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        labels = read_label_file(labels_path)

        camera.resolution = (cfg.CAM_W, cfg.CAM_H)
        camera.framerate = 0.5
        raw_capture = PiRGBArray(camera)

        # image counter for testing
        counter = 0

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                    raw_capture, format="rgb", use_video_port=True
            ):
                raw_capture.truncate(0)
                image = frame.array

                _, scale = set_resized_input(
                    interpreter,
                    (cfg.CAM_W, cfg.CAM_H),
                    lambda size: cv2.resize(image, size),
                )
                interpreter.invoke()

                # get detections
                detections = set()
                objects = get_objects(interpreter, 0.75, scale)

                for obj in objects:
                    if labels and obj.id in labels:
                        label_name = labels[obj.id]
                        detections.add(label_name)

                        # save images for testing
                        # top-left corner
                        tl_corner = (obj.bbox.xmin, obj.bbox.ymin)

                        # bottom-right corner
                        br_corner = (obj.bbox.xmax, obj.bbox.ymax)

                        color = tuple(np.random.random(size=3) * 256)
                        thickness = 2

                        image = cv2.rectangle(image, tl_corner, br_corner, color, thickness)
                        cv2.putText(image, "{0}({1:.2f})".format(label_name, obj.score), tl_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.imwrite("{}.jpg".format(counter), image)
                        counter += 1

                if len(detections) == 0:
                    output = "I see a: nothing :("
                else:
                    output = "I see a: "
                    for d in detections:
                        output += d + " "

                with open("../stylesense/detections.tmp", "w") as detections_file:
                    detections_file.write(output)
                os.rename("../stylesense/detections.tmp", "../stylesense/detections.txt")

        finally:
            pass


if __name__ == "__main__":
    main()
