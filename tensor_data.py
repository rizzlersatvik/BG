import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to get image and mask file paths
def get_data_paths(images_dir, masks_dir):
    image_files = sorted([os.path.join(images_dir, fname) for fname in os.listdir(images_dir)])
    mask_files = sorted([os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir)])
    return image_files, mask_files

# Function to load and preprocess a single image and mask
def load_image_and_mask(image_path, mask_path, target_size=(256, 256)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image to [0, 1]

    mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
    mask = img_to_array(mask)
    mask = mask / 255.0  # Normalize the mask to [0, 1]
    mask = np.round(mask)  # Ensure mask is binary (0 or 1)

    return image, mask

# Function to create TensorFlow dataset
def get_dataset(image_files, mask_files, batch_size=8):
    def generator():
        for image_path, mask_path in zip(image_files, mask_files):
            yield load_image_and_mask(image_path, mask_path)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
        )
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
