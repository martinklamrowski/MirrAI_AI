import tensorflow as tf
import numpy as np
import os

from util import config

# TODO : still working on this, need tf nightly 2 allegedly

# def representative_data_gen():
#     filename_dataset = tf.data.Dataset.list_files(config.TEST_IMAGES_DIR + "*.jpg")
#     print("OVERHEREEEERERERERE")
#     print(tf.rank(filename_dataset))
#     # for f in filename_dataset.take(5):
#     #     print(f.numpy())
#     # filename_dataset = filename_dataset.apply(tf.contrib.data.unbatch())
#     image_dataset = filename_dataset.map(parse_image)
#
#     for input_value in tf.data.Dataset.from_tensor_slices(image_dataset).batch(1).take(100):
#         yield [input_value]
#
#
# def parse_image(filename):
#     parts = tf.strings.split(filename, os.sep, result_type="RaggedTensor")
#     label = parts[-2]
#
#     image = tf.io.read_file(filename)
#     image = tf.image.decode_jpeg(image)
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, [config.MODEL_SIZE, config.MODEL_SIZE])
#     return image, label


# tf.enable_eager_execution()

def representative_data_gen():
    for _ in range(250):
        yield [np.random.uniform(0.0, 1.0, size=(1, 416, 416, 3)).astype(np.float32)]


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
converter.experimental_new_converter = False

tflite_model_quant = converter.convert()

with open(config.EDGETPU_FILE, "wb") as f:
    f.write(tflite_model_quant)
