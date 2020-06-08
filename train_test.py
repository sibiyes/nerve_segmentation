import os
import sys
import numpy as np

from shutil import copyfile

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

def file_copy(filenames, src_dir, dest_dir):
    for fname in filenames:
        copyfile(src_dir + '/' + fname, dest_dir + '/' + fname)

def samplpe():
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_jpeg'
    train_data_dir = base_dir + '/ultrasound-nerve-segmentation/train_data_sample'
    test_data_dir = base_dir + '/ultrasound-nerve-segmentation/test_data_sample'
    
    image_mask_files = [x for x in os.listdir(data_dir) if 'mask' in x]
    np.random.shuffle(image_mask_files)
    
    train_mask_files = image_mask_files[:500]
    test_mask_files = image_mask_files[500:750]
    
    # train_mask_files = image_mask_files[:10]
    # test_mask_files = image_mask_files[10:20]
    
    
    train_images_files = [x.replace('_mask', '') for x in train_mask_files]
    test_images_files = [x.replace('_mask', '') for x in test_mask_files]
    
    print(train_mask_files)
    print(train_images_files)
    print('--------------')
    print(test_mask_files)
    print(test_images_files)
    
    
    file_copy(train_images_files, data_dir, train_data_dir)
    file_copy(train_mask_files, data_dir, train_data_dir)
    file_copy(test_images_files, data_dir, test_data_dir)
    file_copy(test_mask_files, data_dir, test_data_dir)
    
def sample2(n):
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_jpeg'
    train_data_dir = base_dir + '/ultrasound-nerve-segmentation/train_data_sample_{}'.format(n)
    test_data_dir = base_dir + '/ultrasound-nerve-segmentation/test_data_sample'
    
    if (not os.path.exists(train_data_dir)):
        os.makedirs(train_data_dir)
    
    image_mask_files = [x for x in os.listdir(data_dir) if 'mask' in x]
    test_image_mask_files = [x for x in os.listdir(test_data_dir) if 'mask' in x]
    
    unused = list(set(image_mask_files) - set(test_image_mask_files))
    np.random.shuffle(unused)
    
    train_mask_files = unused[:n]
    train_images_files = [x.replace('_mask', '') for x in train_mask_files]
    
    print(train_mask_files)
    print(train_images_files)
    
    file_copy(train_images_files, data_dir, train_data_dir)
    file_copy(train_mask_files, data_dir, train_data_dir)
    
    
def main():
    sample2(1500)
    
    
if __name__:
    main()
