import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-t", "--test", required=False, action="store_true", help="If set detections will be saved.")
args = vars(ap.parse_args())


print(args["test"])


