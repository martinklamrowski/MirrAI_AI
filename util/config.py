""" CONFIG CONSTANTS"""

""" MODEL PARAMETERS """
MODEL_SIZE = 416
DATA_FORMAT = "NHWC"
TINY = False
SPP = False
GPU_MEMORY_FRACTION = 1.0
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

""" DIRECTORIES """
OUTPUT_DIR = "res/output/"
TEST_IMAGES_DIR = "res/test_images/"

""" DATA SETS """
CHICTOPIA_DIR = "res/datasets/chictopia/"
MODANET_DIR = "res/datasets/modanet/"
STYLESENSE_DIR = "res/datasets/stylesense/"

""" FILE NAMES """
CHECKPOINT_FILE = "res/checkpoints/model.ckpt"
PB_FILE = "res/yolo_v3.pb"
WEIGHTS_FILE = "res/yolo_v3.weights"
TFLITE_FILE = "res/yolo_v3.tflite"
EDGETPU_FILE = "res/yolo_v3_quantized.tflite"
