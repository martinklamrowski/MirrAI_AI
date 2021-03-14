import config.config as cfg

import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

SUB_KEY = "70fc8781852e4d5e8db75ebea12ddba8"  # TODO : Ya we should probably hide this. Bing auth?
SUB_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"

while True:
    print("Please enter a search term >>> ")
    search_term = str(input())
    offset_count = 0
    total_matches_for_previous_query = 1000

    headers = {"Ocp-Apim-Subscription-Key": SUB_KEY}
    params = {
            "q": search_term,
            "license": "all",
            "imageType": "photo",
            "cc": "CA",
            "count": 32,
            "aspect": "tall",
            "offset": offset_count * 32 if offset_count * 32 < total_matches_for_previous_query else 0
        }

    response = requests.get(SUB_ENDPOINT, headers=headers, params=params)
    response.raise_for_status()
    image_results = response.json()

    thumbnails = [img["thumbnailUrl"] for img in image_results["value"][:32]]

    f, axes = plt.subplots(4, 8)
    for i in range(4):
        for j in range(8):
            image_data = requests.get(thumbnails[i + 4 * j])
            image_data.raise_for_status()
            image = Image.open(BytesIO(image_data.content))
            axes[i][j].imshow(image)
            axes[i][j].axis("off")
    plt.show()
