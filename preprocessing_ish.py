import glob
import os

import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone import ViewField
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from util import config

# arrange data in yolo format ish
# with open(config.STYLESENSE_DIR + "images.txt", "w") as file:
#     for name in glob.glob(config.STYLESENSE_DIR + "data/*.jpg"):
#
#         name = os.path.split(name)[1]
#         file.write("data/" + name + "\n")


# fiftyone ish

# loading set
name = "stylesense"
data_set = fo.Dataset.from_dir(
    config.STYLESENSE_DIR, fo.types.YOLODataset, name=name
)

print(data_set)
print(data_set.head())
# print(foz.list_zoo_models())

# computing hashes
for sample in data_set:
    sample["file_hash"] = fou.compute_filehash(sample.filepath)
    sample.save()

# find them dupes oot!
n_file_hashes = Counter(sample.file_hash for sample in data_set)
file_hashes = [k for k, v in n_file_hashes.items() if v > 1]

view = (data_set
        .match(ViewField("file_hash").is_in(file_hashes))
        .sort_by("file_hash")
        )

# take them dupes oot!
print("Pre-delete len >> {}".format(len(data_set)))

file_hashes_set = set()
for sample in view:
    if sample.file_hash not in file_hashes_set:
        file_hashes_set.add(sample.file_hash)
    else:
        del data_set[sample.id]

print("Post-delete len >> {}".format(len(data_set)))
data_set.export(export_dir=config.STYLESENSE_DIR + "export/", dataset_type=fo.types.YOLODataset)

session = fo.launch_app(dataset=data_set, desktop=False, port=5151)
session.view = view
session.wait()
