import numpy as np


def load_image_into_numpy_array(array):
    return np.array(array)


def get_detections_dict(detector_output):
    num_detections = int(detector_output.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detector_output.items()}
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    return detections


def read_label_file(path):
    labels_dict = dict()
    try:
        with open(path, "r") as label_file:
            lines = label_file.readlines()

            for i in range(0, len(lines), 4):
                labels_dict[
                    int(lines[i + 2].split(":")[1].strip('\n'))] = lines[i + 1].split(":")[1].strip('"\n')
    finally:
        labels_dict = dict()

    return labels_dict
