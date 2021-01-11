import tensorflow as tf

from util import config


def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]


converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=config.PB_FILE,
    input_arrays=["inputs"],
    input_shapes={"inputs": [None, config.MODEL_SIZE, config.MODEL_SIZE, 3]},
    output_arrays=["output_boxes"]
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

with open(config.TFLITE_FILE, "wb") as f:
    f.write(tflite_model)
