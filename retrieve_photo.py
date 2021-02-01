import io
import time
import lmdb
import sqlite3
import pandas as pd
from PIL import Image

from util import config
from util.photo_data import PhotoData

db = sqlite3.connect("file:{}chictopia.sqlite3?mode=ro".format(config.CHICTOPIA_DIR), uri=True)

photos = pd.read_sql("""
    SELECT
        *,
        'http://images2.chictopia.com/' || path AS url
    FROM photos
    WHERE photos.post_id IS NOT NULL AND file_file_size IS NOT NULL
""", con=db)
print("photos = %d" % (len(photos)))
print(photos.head())


p_data = PhotoData("{}photos.lmdb".format(config.CHICTOPIA_DIR))
print(len(p_data))

# first batch is the first 500 images
batch = "batch5/"
batch_num = 5

for i in range(2000, 2500):
    photo = photos.iloc[i]
    print(photo.id)
    print(photo.url)
    p_data[photo.id].save(config.STYLESENSE_DIR + "images/" + batch + photo.url.rsplit("/", 1)[-1])
    # time.sleep(10)

# for i in range(10):
#     photo = photos.iloc[i]
#     print(photo.id)
#     print(photo.url)
#     p_data[photo.id].show()
#     time.sleep(10)
