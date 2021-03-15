import time
import cv2
import os
import argparse
import requests
from io import BytesIO

import picamera
from picamera.array import PiRGBArray

from pycoral.adapters.common import set_resized_input
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

import numpy as np

import config.config as cfg

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--test", required=False, action="store_true", help="If set detections will be saved.")
args = vars(ap.parse_args())

previous_query = ""
offset_count = 0
total_matches_for_previous_query = -1


def run_bing_image_search(query):
    headers = {"Ocp-Apim-Subscription-Key": cfg.BING_SUB_KEY}

    global previous_query
    global offset_count
    global total_matches_for_previous_query

    if previous_query == query:
        offset_count += 1
    else:
        offset_count = 0
        total_matches_for_previous_query = -1

    params = {
        "q": query,
        "license": "all",
        "imageType": "photo",
        "cc": "CA",
        "count": 24,
        "aspect": "tall",
        "offset": offset_count * 24 if offset_count * 24 < total_matches_for_previous_query else 0
    }

    response = requests.get(cfg.BING_SUB_ENDPOINT, headers=headers, params=params)
    response.raise_for_status()
    image_results = response.json()

    thumbnails = [img["thumbnailUrl"] for img in image_results["value"][:35]]
    total_matches_for_previous_query = image_results["totalEstimatedMatches"]

    # TODO : REMOVE.
    with open("../websearchurl.txt", "w") as url_file:
        url_file.write(image_results["webSearchUrl"])

    if len(thumbnails) > 0:
        for i, t in enumerate(thumbnails):
            image_data = requests.get(t)
            image_data.raise_for_status()
            file_bytes = np.asarray(bytearray(BytesIO(image_data.content).read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            cv2.imwrite(cfg.PATH_TO_INSPIRATIONS_IMAGES + "{}.jpg".format(i + 1), image)


def generate_image_search_query(detections_set):
    # TODO : Going to be man by default for now.
    man = True
    # TODO : Clean this up damn.
    relevant_detections = {"suit jacket", "dress shirt", "long-sleeve", "long coat",
                           "cardigan", "short-sleeve", "jean jacket", "winter jacket",
                           "tank-top", "shorts", "athletic pants", "pants"}
    query = ""
    if man:
        query = "mens outfits with "
        for det in detections_set:
            if det in relevant_detections:
                if det == "short-sleeve":
                    query += "t-shirt "
                elif det == "dress shirt":
                    return "mens outfits with dress shirt"
                else:
                    query += det + " "

    if query == "mens outfits with ":
        return ""
    return query


def poll_trigger_file():
    with open(cfg.PATH_TO_SHARED_FILES + "triggerfile.txt", "r") as trigger_file:
        if trigger_file.read() == "TRUE":
            return True

    return False


def reset_trigger_file():
    with open(cfg.PATH_TO_SHARED_FILES + "triggerfile.tmp", "w") as trigger_file:
        trigger_file.write("FALSE")

    os.rename(cfg.PATH_TO_SHARED_FILES + "triggerfile.tmp", cfg.PATH_TO_SHARED_FILES + "triggerfile.txt")


def main():
    model_path = cfg.PATH_TO_STYLESENSE_MODEL
    labels_path = cfg.PATH_TO_STYLESENSE_LABELS

    with picamera.PiCamera() as camera:

        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        labels = read_label_file(labels_path)

        camera.resolution = (cfg.CAM_W, cfg.CAM_H)
        camera.framerate = 10
        raw_capture = PiRGBArray(camera)

        # image counter for testing
        counter = 0

        # allow the camera to warmup
        time.sleep(1)

        try:
            for frame in camera.capture_continuous(
                    raw_capture, format="rgb", use_video_port=True
            ):

                raw_capture.truncate(0)

                # colours are incorrectly mapped; images are blue without this line
                image = cv2.cvtColor(frame.array, cv2.COLOR_BGR2RGB)

                if poll_trigger_file():
                    _, scale = set_resized_input(
                        interpreter,
                        (cfg.CAM_W, cfg.CAM_H),
                        lambda size: cv2.resize(image, size),
                    )
                    interpreter.invoke()

                    # get detections
                    detections = set()
                    objects = get_objects(interpreter, 0.50, scale)

                    for obj in objects:
                        if labels and obj.id in labels:
                            label_name = labels[obj.id]
                            detections.add(label_name)

                            if args["test"]:
                                # save images for testing
                                # top-left corner
                                tl_corner = (obj.bbox.xmin, obj.bbox.ymin)

                                # bottom-right corner
                                br_corner = (obj.bbox.xmax, obj.bbox.ymax)

                                color = tuple(np.random.random(size=3) * 256)
                                thickness = 2

                                image = cv2.rectangle(image, tl_corner, br_corner, color, thickness)
                                cv2.putText(image, "{0}({1:.2f})".format(label_name, obj.score), tl_corner,
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                                cv2.imwrite("{}{}.jpg".format(cfg.PATH_TO_TESTING_OUTPUT, counter), image)
                                counter += 1

                    if len(detections) == 0:
                        output = "I see a: nothing :("
                    else:
                        output = "I see a: "
                        for d in detections:
                            output += d + " "

                    if args["test"]:
                        # also testing
                        print(output)

                    # for stylesense
                    with open(cfg.PATH_TO_SHARED_FILES + "detections.tmp", "w") as detections_file:
                        detections_file.write(output)
                    os.rename(cfg.PATH_TO_SHARED_FILES + "detections.tmp", cfg.PATH_TO_SHARED_FILES + "detections.txt")

                    # for style variations
                    query = generate_image_search_query(detections)

                    if query != "":
                        run_bing_image_search(query)

                    # don't poll again for at least 5 seconds (no spam please!)
                    reset_trigger_file()
                    time.sleep(5)

        finally:
            pass


if __name__ == "__main__":
    main()
