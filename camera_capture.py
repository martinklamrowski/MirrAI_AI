import cv2
import utils.config as config


def get_capture():
    camera = cv2.VideoCapture(config.DEVICE_ID)

    ret, image = camera.read()
    cv2.imwrite(config.PATH + "selfie.png", image)

    return config.PATH + "selfie.png"
