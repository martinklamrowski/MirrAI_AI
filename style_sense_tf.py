import time
import os

import picamera
from picamera.array import PiRGBArray

import tensorflow as tf

import config.config as cfg
import utils.utils as uti


def main():
    detector = tf.saved_model.load(cfg.PATH_TO_PB_MODEL)

    with picamera.PiCamera() as camera:
        labels = uti.read_label_file(cfg.PATH_TO_LABELS)

        camera.resolution = (cfg.CAM_W, cfg.CAM_H)
        camera.framerate = 10
        raw_capture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                    raw_capture, format="rgb", use_video_port=True
            ):
                time.sleep(2)

                raw_capture.truncate(0)
                image_np = uti.load_image_into_numpy_array(frame.array)
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]

                # get detections
                detections_dict = uti.get_detections_dict(detector(input_tensor))
                objects_set = set()

                # for det in detections_dict:
                #     if labels and det.id in labels:
                #         label_name = labels[det.id]
                #         objects_set.add(label_name)
                print(detections_dict)

                if len(objects_set) == 0:
                    output = "I see a: nothing :("
                else:
                    output = "I see a: "
                    for o in objects_set:
                        output += o + " "

                print(output)
                with open("../stylesense/detections.tmp", "w") as detections_file:
                    detections_file.write(output)
                os.rename("../stylesense/detections.tmp", "../stylesense/detections.txt")

        finally:
            pass


if __name__ == "__main__":
    main()
