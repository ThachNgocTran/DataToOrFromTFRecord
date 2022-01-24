# Source: https://keras.io/examples/keras_recipes/creating_tfrecords/

import os, json
import tensorflow as tf

root_dir = r'./datasets'
tfrecords_dir = r'./tfrecords'
images_dir = os.path.join(root_dir, "val2017")
annotations_dir = os.path.join(root_dir, "annotations")
annotation_file = os.path.join(annotations_dir, "instances_val2017.json")
images_url = "http://images.cocodataset.org/zips/val2017.zip"
annotations_url = ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip")

# Download image files if not yet.
if not os.path.exists(images_dir):
    image_zip = tf.keras.utils.get_file(
        "images.zip",
        cache_dir=os.path.abspath("."),
        cache_subdir="datasets",
        origin=images_url,
        extract=True
    )
    os.remove(image_zip)

# Download caption annotation files if not yet.
if not os.path.exists(annotations_dir):
    annotation_zip = tf.keras.utils.get_file(
        "captions.zip",
        cache_dir=os.path.abspath("."),
        cache_subdir="datasets",
        origin=annotations_url,
        extract=True
    )
    os.remove(annotation_zip)

with open(annotation_file, "r", encoding="utf-8") as f:
    annotations = json.load(f)["annotations"]

# Number of samples in one TFRecord file.
num_samples = 4096

# Number of TFRecord files to be created.
num_tfrecords = len(annotations) // num_samples
if len(annotations) % num_samples:
    num_tfrecords += 1

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)

# Convert a 3-D uint to a Feature of "bytes_list", aka. [bytes, bytes, bytes...]
def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]) # JPG Quality 95 by default.
    )

# Convert a string to a Feature of "bytes_list", aka. [bytes, bytes, bytes...]
def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

# Convert a float to a Feature of "float_list", aka. [float, float, float...]
def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# Convert an int to a Feature of "int64_list", aka. [int64, int64, int64...]
def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Convert a list of floats to a Feature of "float_list", aka. [float, float, float...]
def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, example):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "area": float_feature(example["area"]),
        "bbox": float_feature_list(example["bbox"]),
        "category_id": int64_feature(example["category_id"]),
        "id": int64_feature(example["id"]),
        "image_id": int64_feature(example["image_id"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

for tfrec_num in range(num_tfrecords):
    samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]

    with tf.io.TFRecordWriter(
            tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
    ) as writer:
        for sample in samples:
            image_path = f"{images_dir}/{sample['image_id']:012d}.jpg"
            # Try to decode the JPG file => check if well-formed.
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            example = create_example(image, image_path, sample)
            writer.write(example.SerializeToString())
