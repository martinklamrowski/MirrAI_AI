import fiftyone as fo
from util import config
import glob
import os


# arrange data in yolo format ish
# with open(config.STYLESENSE_DIR + "images.txt", "w") as file:
#     for name in glob.glob(config.STYLESENSE_DIR + "data/*.jpg"):
#
#         name = os.path.split(name)[1]
#         file.write("data/" + name + "\n")


# fiftyone ish
name = "stylesense"
data_set = fo.Dataset.from_dir(
    config.STYLESENSE_DIR, fo.types.YOLODataset, name=name
)

print(data_set)
print(data_set.head())

session = fo.launch_app(dataset=data_set, desktop=False, port=5151)
session.wait()
