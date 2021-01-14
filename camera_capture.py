import cv2
import util.config as config


def get_capture():
    camera = cv2.VideoCapture(0)

    ret, image = camera.read()
    cv2.imwrite(config.OUTPUT_DIR + "selfie.png", image)

    return config.OUTPUT_DIR + "selfie.png"
