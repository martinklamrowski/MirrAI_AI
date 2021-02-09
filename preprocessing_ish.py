import glob
import os
import json
from tqdm import tqdm

import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone import ViewField
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from util import config

SS_16_TO_10_MAP = {
    "0": "0", "1": "1", "2": "6", "3": "8", "4": "8", "5": "3", "6": "4",
    "7": "5", "8": "6", "9": "2", "10": "7", "11": "3", "12": "4", "13": "9",
    "14": "2", "15": "6"
}

# arrange data in yolo format ish
# with open(config.STYLESENSE_DIR + "data/images.txt", "w") as file:
#     for name in glob.glob(config.STYLESENSE_DIR + "data/images/*.jpg"):
#
#         name = os.path.split(name)[1]
#         file.write("data/images/" + name + "\n")


# stealing modanet annotations
# moda_json_files = glob.glob(config.MODANET_DIR + "data/*.json")
#
# for json_file in tqdm(moda_json_files):
#     with open(json_file) as data_file:
#         moda_data = json.load(data_file)
#         # for item in moda_data.values():
#         #     print(item)

# convert existing stylesense_16 annotations to stylesense_10
for an_file in glob.glob(config.STYLESENSE_DIR + "data/images/*.txt"):
    with open(an_file, "r") as data_file:
        contents = [line.split(" ") for line in data_file.readlines()]
        for entry in contents:
            entry[0] = SS_16_TO_10_MAP[entry[0]]

    with open(an_file, "w") as data_file:
        for entry in contents:
            data_file.write(" ".join(entry))

    print(contents)
    print("FUCKYOU")




# fiftyone ish

# # loading set
# name = "stylesense"
# data_set = fo.Dataset.from_dir(
#     config.STYLESENSE_DIR, fo.types.YOLODataset, name=name
# )
#
# print(data_set)
# print(data_set.head())
# # print(foz.list_zoo_models())
#
# # computing hashes
# for sample in data_set:
#     sample["file_hash"] = fou.compute_filehash(sample.filepath)
#     sample.save()
#
# # find them dupes oot!
# n_file_hashes = Counter(sample.file_hash for sample in data_set)
# file_hashes = [k for k, v in n_file_hashes.items() if v > 1]
#
# view = (data_set
#         .match(ViewField("file_hash").is_in(file_hashes))
#         .sort_by("file_hash")
#         )
#
# # take them dupes oot!
# print("Pre-delete len >> {}".format(len(data_set)))
#
# file_hashes_set = set()
# for sample in view:
#     if sample.file_hash not in file_hashes_set:
#         file_hashes_set.add(sample.file_hash)
#     else:
#         del data_set[sample.id]
#
# print("Post-delete len >> {}".format(len(data_set)))
# data_set.export(export_dir=config.STYLESENSE_DIR + "export/", dataset_type=fo.types.YOLODataset)
#
# session = fo.launch_app(dataset=data_set, desktop=False, port=5151)
# session.view = view
# session.wait()
