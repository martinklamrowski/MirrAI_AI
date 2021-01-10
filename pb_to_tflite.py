import tensorflow as tf

from util import config


converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=config.PB_FILE,
    input_arrays=["inputs"],
    input_shapes={"inputs": [None, config.MODEL_SIZE, config.MODEL_SIZE, 3]},
    output_arrays=["output_boxes"]
)

tflite_model = converter.convert()

with open(config.TFLITE_FILE, "wb") as f:
    f.write(tflite_model)
