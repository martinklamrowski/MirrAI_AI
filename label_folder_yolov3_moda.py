import torch
import glob
import json
import cv2

from predictors.YOLOv3 import YOLOv3Predictor
from yolo.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

"""
Script to label all images in a directory. Writes list of detections to 
a .json with detections mapped to image name.
"""

yolo_moda_params = {"model_def": "yolo/modanetcfg/yolov3-modanet.cfg",
                    "weights_path": "yolo/weights/yolov3-modanet_last.weights",
                    "class_path": "yolo/modanetcfg/modanet.names",
                    "conf_thres": 0.5,
                    "nms_thres": 0.4,
                    "img_size": 416,
                    "device": device}

yolo_params = yolo_moda_params
model = YOLOv3Predictor(params=yolo_params)
classes = utils.load_classes(yolo_params["class_path"])
dataset = "moda"

out = {"images": []}
images = [image for image in glob.iglob(r"images_to_label/*.*")]

print("found {} images in 'images_to_label/'".format(len(images)))


for path in images:
    image = cv2.imread(path)
    detections = model.get_detections(image)
    print(detections)
    for x1, y1, x2, y2, cls_conf, cls_pred in detections:
        print("\t+ Label: {}, Conf: {:.5f}".format(classes[int(cls_pred)], cls_conf))

    print(path)
    out["images"].append({
        "file": path.split("/")[1],
        "labels": [classes[int(d)] for _, _, _, _, _, d in detections]
    })

with open("images_to_label/labelled_images.json", "w") as outfile:
    json.dump(out, outfile)
