import os
import sys

import tensorflow as tf

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import pathlib

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

image_count = 3200
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
def get_label(file_path):
      # convert the path to a list of path components
      parts = tf.strings.split(file_path, os.path.sep)
      # The second to last is the class-directory
      return parts[-2] == CLASS_NAMES

def decode_img(img, num_channels, shape):
      # convert the compressed string to a 3D uint8 tensor
      img = tf.image.decode_jpeg(img, channels = num_channels)
      # Use `convert_image_dtype` to convert to floats in the [0,1] range.
      img = tf.image.convert_image_dtype(img, tf.float32)
      # resize the image to the desired size.
      return tf.image.resize(img, shape)

def process_path(file_path, shape = None):
      # load the raw data from the file as a string
      #shape = [128, 128]
      image_path = tf.strings.regex_replace(file_path, '_mask', '')
      img = tf.io.read_file(image_path)
      img_mask = tf.io.read_file(file_path)
      img = decode_img(img, 3, shape)
      img_mask = decode_img(img_mask, 1, shape)
      return img, img_mask
      
      
def load_data(data_dir, shape, n = None):
    ### https://www.tensorflow.org/tutorials/load_data/images
    ### https://stackoverflow.com/questions/58185222/read-image-and-mask-for-segmentation-problem-in-tensorflow-2-0-using-tf-data#
    
    data_dir = pathlib.Path(data_dir)
    
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*mask.jpeg'))
    if (n is not None):
        list_ds = list_ds.take(n)
    
    # fn = lambda f: process_path(shape)
    # labeled_ds = list_ds.map(fn, num_parallel_calls=AUTOTUNE)
    labeled_ds = list_ds.map(lambda x: process_path(x, shape), num_parallel_calls=AUTOTUNE)
    
    
    return labeled_ds
    
    
def main():
    load_data()
    
if __name__ == '__main__':
    main()
