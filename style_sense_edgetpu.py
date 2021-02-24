import time
import cv2
import os

import picamera
from picamera.array import PiRGBArray

from pycoral.adapters.common import set_resized_input
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

import config.config as cfg


def main():
    default_model = "../stylesense/data/mobilenetv2_stylesense16_quant_int8_edgetpu.tflite"
    default_labels = "../stylesense/data/stylesense_labels.txt"

    with picamera.PiCamera() as camera:

        interpreter = make_interpreter(default_model)
        interpreter.allocate_tensors()
        labels = read_label_file(default_labels)

        camera.resolution = (cfg.CAM_W, cfg.CAM_H)
        camera.framerate = 10
        raw_capture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                    raw_capture, format="rgb", use_video_port=True
            ):
                # start_ms = time.time()
                time.sleep(2)

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
                objects = get_objects(interpreter, 0.5, scale)

                for obj in objects:
                    if labels and obj.id in labels:
                        label_name = labels[obj.id]
                        detections.add(label_name)
                        # caption = "{0}({1:.2f})".format(label_name, obj.score)

                        # bbox = (
                        #     obj.bbox.xmin,
                        #     obj.bbox.ymin,
                        #     obj.bbox.xmax,
                        #     obj.bbox.ymax,
                        # )

                        # print("{} - BBox: {}".format(caption, bbox))
                        # print("{}".format(caption))
                    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if len(detections) == 0:
                    # print("I see a: nothing :(", file=sys.stdout)
                    output = "I see a: nothing :("
                else:
                    output = "I see a: "
                    for d in detections:
                        output += d + " "
                    # print(output, file=sys.stdout)
                # fps ish
                # elapsed_ms = time.time() - start_ms
                # fps = 1 / elapsed_ms
                # fps_text = "{0:.2f}ms, {1:.2f}fps".format((elapsed_ms * 1000.0), fps)
                # print(fps_text)

                with open("../stylesense/detections.tmp", "w") as detections_file:
                    detections_file.write(output)
                os.rename("../stylesense/detections.tmp", "../stylesense/detections.txt")

        finally:
            # with open("detections.txt", "w") as detections_file:
            #     # just empty the file on quit
            #     detections_file.write(":)")
            pass


if __name__ == "__main__":
    main()
