import requests
import json
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def print_json(obj):
    print(json.dumps(obj, sort_keys=True, indent=2, separators=(",", ":")))


SUB_KEY = "70fc8781852e4d5e8db75ebea12ddba8"  # TODO : Ya we should probably hide this. Bing auth?
SUB_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
HEADERS = {"Ocp-Apim-Subscription-Key": SUB_KEY}


while True:
    print("Please enter an image path >>> ")
    image_path = str(input())

    file = {"image": ("myfile", open(image_path, "rb"))}

    response = requests.post(SUB_ENDPOINT, headers=HEADERS, files=file)
    response.raise_for_status()

    results = response.json()
    print_json(results)

    print(results.keys())
    print("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}")
    print(results["tags"])

    tags = results["tags"]

    fig = plt.figure(figsize=(13, 5))
    ax = []

    # this is some deep ass json
    for i in range(len(tags)):
        tag = tags[i]
        actions = tag["actions"]

        for j in range(len(actions)):
            action = actions[j]
            if action["actionType"] == "VisualSearch":
                data = action["data"]
                num_items = data["totalEstimatedMatches"]
                items = data["value"]

                # for k in range(num_items): # there are alot of items; lets just take 24 for now
                for k in range(24):
                    # phew
                    thumbnail_url = items[k]["thumbnailUrl"]
                    image_data = requests.get(thumbnail_url)
                    image_data.raise_for_status()
                    image = Image.open(BytesIO(image_data.content))
                    ax.append((fig.add_subplot(8, 3, k + 1, )))
                    ax[-1].axis("off")
                    plt.imshow(image)
                break
        break
    plt.show()





    # f, axes = plt.subplots(4, 8)
    # for i in range(4):
    #     for j in range(8):
    #         image_data = requests.get(thumbnails[i + 4 * j])
    #         image_data.raise_for_status()
    #         image = Image.open(BytesIO(image_data.content))
    #         axes[i][j].imshow(image)
    #         axes[i][j].axis("off")
    # plt.show()
