# these paths won't work if using start.sh
PATH_TO_PB_MODEL = "data/saved_model"
PATH_TO_LABELS = "data/label_map.pbtxt"
PATH_TO_TESTING_OUTPUT = "data/testing_output/"

CAM_W = 1080
CAM_H = 1080

BING_SUB_KEY = "70fc8781852e4d5e8db75ebea12ddba8"  # TODO : Ya we should probably hide this. Bing auth?
BING_IMAGE_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"
BING_VISUAL_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/visualsearch"

PATH_TO_SHARED_FILES = "../shared_files/"
PATH_TO_INSPIRATIONS_IMAGES = "../MirrAI_UI/modules/StyleInspirations/images/"
PATH_TO_VARIATIONS_IMAGES = "../MirrAI_UI/modules/StyleVariations/images/"
PATH_TO_STYLESENSE_MODEL = "../stylesense/data/ssd_mobilenet_v2_stylesense16_quant_postprocess_edgetpu.tflite"
PATH_TO_STYLESENSE_LABELS = "../stylesense/data/stylesense_labels.txt"
PATH_TO_COCO_MODEL = "../stylesense/data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
PATH_TO_COCO_LABELS = "../stylesense/data/coco_labels.txt"
