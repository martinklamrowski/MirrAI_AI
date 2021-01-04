import os
import matplotlib.pyplot as plt
import cv2

from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# YOLO PARAMS
yolo_df2_params = {"model_def": "yolo/df2cfg/yolov3-df2.cfg",
                   "weights_path": "yolo/weights/yolov3-df2_15000.weights",
                   "class_path": "yolo/df2cfg/df2.names",
                   "conf_thres": 0.5,
                   "nms_thres": 0.4,
                   "img_size": 416,
                   "device": device}

yolo_modanet_params = {"model_def": "yolo/modanetcfg/yolov3-modanet.cfg",
                       "weights_path": "yolo/weights/yolov3-modanet_last.weights",
                       "class_path": "yolo/modanetcfg/modanet.names",
                       "conf_thres": 0.5,
                       "nms_thres": 0.4,
                       "img_size": 416,
                       "device": device}

# DATASET
# yolo_params = yolo_df2_params
# dataset = "df"
yolo_params = yolo_modanet_params
dataset = "moda"

# Classes
classes = load_classes(yolo_params["class_path"])

# Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
# np.random.shuffle(colors)

model = YOLOv3Predictor(params=yolo_params)

while True:
    path = input("path >>> ")
    if not os.path.exists(path):
        print("you suck")
        continue
    img = cv2.imread(path)
    detections = model.get_detections(img)
    # print(detections)

    # unique_labels = np.array(list(set([det[-1] for det in detections])))

    # n_cls_preds = len(unique_labels)
    # bbox_colors = colors[:n_cls_preds]

    if len(detections) != 0:
        detections.sort(reverse=False, key=lambda x: x[4])
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))

            # color = bbox_colors[np.where(unique_labels == cls_pred)[0]][0]
            color = colors[int(cls_pred)]

            color = tuple(c * 255 for c in color)
            color = (0.7 * color[2], 0.7 * color[1], 0.7 * color[0])

            font = cv2.FONT_HERSHEY_SIMPLEX

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            text = "%s conf: %.3f" % (classes[int(cls_pred)], cls_conf)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            y1 = 0 if y1 < 0 else y1
            y1_rect = y1 - 25
            y1_text = y1 - 5

            if y1_rect < 0:
                y1_rect = y1 + 27
                y1_text = y1 + 20
            cv2.rectangle(img, (x1 - 2, y1_rect), (x1 + int(8.5 * len(text)), y1), color, -1)
            cv2.putText(img, text, (x1, y1_text), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    img_resized = cv2.resize(img, (1000, 1300), interpolation=cv2.INTER_AREA)

    cv2.imshow("boundings", img_resized)
    img_id = path.split('/')[-1].split('.')[0]
    cv2.imwrite("output/ouput-test_{}_yolov3_{}.jpg".format(img_id, model, dataset), img_resized)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
