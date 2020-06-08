import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import pathlib

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# IMG_WIDTH = 224
# IMG_HEIGHT = 224

IMG_WIDTH = 580
IMG_HEIGHT = 420

def decode_img(img, num_channels):
      # convert the compressed string to a 3D uint8 tensor
      img = tf.image.decode_jpeg(img, channels = num_channels)
      # Use `convert_image_dtype` to convert to floats in the [0,1] range.
      img = tf.image.convert_image_dtype(img, tf.float32)
      # resize the image to the desired size.
      return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def image_level_count():
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_jpeg'
    fp = open(base_dir + '/analysis/num_mask_pixels_all.csv', 'w')
    
    mask_files = [f for f in os.listdir(data_dir) if '_mask' in f]
    mask_files = sorted(mask_files, key = lambda x: [int(a) for a in x.split('.')[0].split('_')[:2]])
    
    total_pos_pixels = 0
    total_pixels = 0
    
    i = 0
    for img_f in mask_files:
        print(img_f)
        if ('mask' not in img_f):
            continue
            
        img = tf.io.read_file(data_dir + '/' + img_f)
        img = decode_img(img, 1)
        #print(img.numpy())
        num_pos_pixels = np.sum(img.numpy().astype(int))
        percentage = num_pos_pixels/(IMG_WIDTH*IMG_HEIGHT)
        
        total_pos_pixels += num_pos_pixels
        total_pixels += IMG_WIDTH*IMG_HEIGHT
            
        #sys.exit(0)
        
        line = img_f.split('.')[0] + ',' + str(num_pos_pixels) + ',' + str(percentage)
        fp.write(line + '\n')
        
        i += 1
        
        # if (i >= 500):
        #     break
        
    line = 'total' + ',' + str(total_pos_pixels) + ',' + str(total_pos_pixels/total_pixels)
    fp.write(line + '\n')
        
    fp.close()
    
def stats(image_stats):
    image_stats_vals = image_stats.values
    print('number of images: ', np.shape(image_stats_vals)[0] - 1)
    print('number of images with empty mask: ', np.sum(image_stats_vals[:, 1][:-1] == 0))
    print('percent positive pixels in positive masks: ', np.mean(image_stats_vals[:-1][image_stats_vals[:-1][:, 1] != 0][:, 2])*100)
    print('percent pixels with positive mask: ', image_stats_vals[-1][2]*100)
    
def mask_analysis():
    img_level_count_all_file = base_dir + '/analysis/num_mask_pixels_all.csv'
    img_level_count_train_sample_file = base_dir + '/analysis/num_mask_pixels_train_sample.csv'
    
    fp = open(img_level_count_all_file, 'rb')
    img_level_count_all = pd.read_csv(fp, header = None)
    fp.close()
    
    fp = open(img_level_count_train_sample_file, 'rb')
    img_level_count_train_sample = pd.read_csv(fp, header = None)
    fp.close()
    
    print(img_level_count_all)
    print(img_level_count_train_sample)
    
    stats(img_level_count_all)
    
def prediction_analysis():
    tag = 'resnet_model_s3_b5_relu_w10'
    data_tag = 'test'
    
    prediction_results_dir = base_dir + '/prediction_results/{0}/{1}'.format(tag, data_tag)
    
    
    fp_in = open(prediction_results_dir + '/results_img_0.5.csv', 'rb')
    # img_results = pd.read_csv(fp_in, skiprows = 1, header = None)
    # fp_in.close()
    
    lines = fp_in.readlines()[1:-1]
    lines = [l.decode().strip().split(',') for l in lines]
    fp_in.close()
    #img_results = pd.DataFrame(lines, columns = ['img', 'pos_pixels', 'neg_pixels', 'tp', 'fp', 'tn', 'fn', 'dice', 'iou', 'i1', 'i2'])
    img_results = pd.DataFrame(lines, columns = ['img', 'pos_pixels', 'neg_pixels', 'tp', 'fp', 'tn', 'fn', 'dice', 'iou'])
    
    #img_results.columns = ['img', 'pos_pixels', 'neg_pixels', 'tp', 'fp', 'tn', 'fn', 'dice', 'iou', 'ignore1', 'ignore2']
    
    img_results = img_results.astype({'pos_pixels': float, 'neg_pixels': float, 'tp': float, 'fp': float, 'tn': float, 'fn': float, 'dice': float, 'iou': float})
    img_results = img_results.astype({'pos_pixels': int, 'neg_pixels': int, 'tp': int, 'fp': int, 'tn': int, 'fn': int, 'dice': float, 'iou': float})
    # print(img_results)
    # sys.exit(0)
    
    neg_image_nofp = img_results[(img_results['tp'] + img_results['fn'] == 0) & (img_results['fp'] == 0)]
    neg_image_fp = img_results[(img_results['tp'] + img_results['fn'] == 0) & (img_results['fp'] != 0)]

    print('neg_image_nofp', len(neg_image_nofp))
    print('neg_image_fp', len(neg_image_fp))
    
    pos_image_tp = img_results[(img_results['tp'] + img_results['fn'] != 0) & (img_results['tp'] != 0)]
    pos_image_notp = img_results[(img_results['tp'] + img_results['fn'] != 0) & (img_results['tp'] == 0)]
    
    # print(pos_image_notp)
    # print(pos_image_tp)
    
    print('-------------------')
    
    print('pos_image_tp', len(pos_image_tp))
    print('pos_image_notp', len(pos_image_notp))
    
    #print(img_results[(img_results['tp'] + img_results['fn'] != 0)])
    
    
    
def main():
    #image_level_count()
    #mask_analysis()
    prediction_analysis()
        
        
        
if __name__ == '__main__':
    main()
    
