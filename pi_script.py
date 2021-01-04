import torch
import cv2

import camera_capture as cam
from predictors.YOLOv3 import YOLOv3Predictor
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# get capture
image_path = cam.get_capture()

# make detections
yolo_params = config.YOLO_MODA_PARAMS
dataset = "MODA"
classes = utils.load_classes(yolo_params["class_path"])
model = YOLOv3Predictor(params=yolo_params)

image = cv2.imread(image_path)
detections = model.get_detections(image)
print(detections)

# TODO
# send detections to firebase

# TODO
# wait to receive 4 images from firebase

# TODO
# save images from firebase to "recos" directory
