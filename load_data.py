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

from dataloader.dataloader import DataLoader

def load_data2():
    ### https://github.com/HasnainRaz/SemSegPipeline
    
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_jpeg'
    
    image_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.tif') and 'mask' not in x]
    mask_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.tif') and 'mask' in x]
    
    # print(image_paths)
    # print(mask_paths)
    
    dataset = DataLoader(image_paths=image_paths,
                     mask_paths=mask_paths,
                     image_size=[256, 256],
                     crop_percent=1.0,
                     channels=[3, 3],
                     seed=47)
                     
    dataset = dataset.data_batch(batch_size=BATCH_SIZE,
                             augment=False, 
                             shuffle=True)
                             
    print(dataset)
    
def get_label(file_path):
      # convert the path to a list of path components
      parts = tf.strings.split(file_path, os.path.sep)
      # The second to last is the class-directory
      return parts[-2] == CLASS_NAMES

def decode_img(img, num_channels, img_size):
      # convert the compressed string to a 3D uint8 tensor
      img = tf.image.decode_jpeg(img, channels = num_channels)
      #tf.io.decode_image(img, name=None)
      # Use `convert_image_dtype` to convert to floats in the [0,1] range.
      img = tf.image.convert_image_dtype(img, tf.float32)
      # resize the image to the desired size.
      #return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
      return tf.image.resize(img, img_size)

def process_path(file_path, img_size):
      #label = get_label(file_path)
      # load the raw data from the file as a string
      image_path = tf.strings.regex_replace(file_path, '_mask', '')
      img = tf.io.read_file(image_path)
      img_mask = tf.io.read_file(file_path)
      img = decode_img(img, 3, img_size)
      img_mask = decode_img(img_mask, 1, img_size)
      return img, img_mask
      
      
def load_data():
    ### https://www.tensorflow.org/tutorials/load_data/images
    ### https://stackoverflow.com/questions/58185222/read-image-and-mask-for-segmentation-problem-in-tensorflow-2-0-using-tf-data#
    
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_jpeg'
    data_dir = pathlib.Path(data_dir)
    
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*mask.jpeg'))
    
    for f in list_ds.take(5):
        print(f.numpy())
        
    #labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    labeled_ds = list_ds.map(lambda x: process_path(x, [256, 256]), num_parallel_calls=AUTOTUNE)
    print(labeled_ds)
    
    for image, mask in labeled_ds.take(1):
        # print(image.numpy())
        # print(mask.numpy())
        print(image.numpy().shape)
        print(mask.numpy().shape)
        print(np.unique(np.round(mask.numpy())))
        print(type(image))
        print(type(mask))
        
        print('----------------------')

    
def main():
    load_data()
    
if __name__ == '__main__':
    main()
