import torch
import cv2

import camera_capture as cam
from predictors.YOLOv3 import YOLOv3Predictor
from utils import *

# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db

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

# # Firebase connection
# cred = credentials.Certificate(r"/Users/angie/Downloads/Firebase_Connection/smartmirrai-c2051-firebase-adminsdk-pnq5k-19437db28a.json")
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
