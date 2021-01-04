"""CONFIG CONSTANTS"""
import torch

# CAMERA
DEVICE_ID = 0
PATH = "/captures/"

# YOLO
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_DF_PARAMS = {"model_def": "yolo/df2cfg/yolov3-df2.cfg",
                  "weights_path": "yolo/weights/yolov3-df2_15000.weights",
                  "class_path": "yolo/df2cfg/df2.names",
                  "conf_thres": 0.5,
                  "nms_thres": 0.4,
                  "img_size": 416,
                  "device": DEVICE}

YOLO_MODA_PARAMS = {"model_def": "yolo/modanetcfg/yolov3-modanet.cfg",
                    "weights_path": "yolo/weights/yolov3-modanet_last.weights",
                    "class_path": "yolo/modanetcfg/modanet.names",
                    "conf_thres": 0.5,
                    "nms_thres": 0.4,
                    "img_size": 416,
                    "device": DEVICE}
