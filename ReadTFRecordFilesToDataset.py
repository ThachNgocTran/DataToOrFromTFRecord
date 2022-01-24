# Source: https://keras.io/examples/keras_recipes/creating_tfrecords/

import tensorflow as tf

tfrecords_dir = r'./tfrecords'

def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example

def prepare_sample(features):
    image = tf.image.resize(features["image"], size=(224, 224))
    return image, features["category_id"]

def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
            # Restore the original entry.
            .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
            # Filter interested features.
            .map(prepare_sample, num_parallel_calls=AUTOTUNE)
            .shuffle(batch_size * 10)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
    )
    return dataset

train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")
batch_size = 32
epochs = 1
steps_per_epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

input_tensor = tf.keras.layers.Input(shape=(224, 224, 3), name="image")
model = tf.keras.applications.EfficientNetB0(
    input_tensor=input_tensor, weights=None, classes=91
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Create the dataset.
the_dataset = get_dataset(train_filenames, batch_size)

model.fit(
    x=the_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)
