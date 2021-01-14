from picamera import PiCamera
import time

import util.config as config


def get_capture():

    cam = PiCamera()
    cam.resolution = (1080, 1920)
    time.sleep(0.5)

    cam.capture(config.OUTPUT_DIR + "selfie.jpg")

    return config.OUTPUT_DIR + "selfie.jpg"
