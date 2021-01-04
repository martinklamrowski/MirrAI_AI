import torch
import glob
import json
import cv2

from predictors.YOLOv3 import YOLOv3Predictor
from yolo.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


