import io
import lmdb
from PIL import Image


class PhotoData(object):
    def __init__(self, path):
        self.env = lmdb.open(
            path, map_size=2 ** 36, readonly=True, lock=False
        )

    def __iter__(self):
        with self.env.begin() as t:
            with t.cursor() as c:
                for key, value in c:
                    yield key, value

    def __getitem__(self, index):
        key = str(index).encode("ascii")
        with self.env.begin() as t:
            data = t.get(key)
        if not data:
            return None
        with io.BytesIO(data) as f:
            image = Image.open(f)
            image.load()
            return image

    def __len__(self):
        return self.env.stat()["entries"]
