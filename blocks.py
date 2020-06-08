import os
import sys

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np
import matplotlib.pyplot as plt
import pickle

def downsampling_block(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(conv2)
    
    return pooling, conv2
    
def downsampling_block_dilation(inputs, n_channels, kernel_size, pool_size, activation, dilation_rate, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, dilation_rate = dilation_rate, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, dilation_rate = dilation_rate, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(conv2)
    
    return pooling, conv2
    
def downsampling_block_bn(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv1_bn = tf.keras.layers.BatchNormalization(name = tag + '_conv1_bn')(conv1)
    conv1_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_act')(conv1_bn)
    
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1_act)
    conv2_bn = tf.keras.layers.BatchNormalization(name = tag + '_conv2_bn')(conv2)
    conv2_act = tf.keras.layers.Activation(activation, name = tag + '_conv2_act')(conv2_bn)
    
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(conv2_act)
    
    return pooling, conv2_act
    
def downsampling_block_dropout(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv1_do = tf.keras.layers.SpatialDropout2D(0.1, name = tag + '_conv1_do')(conv1)
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1_do)
    conv2_do = tf.keras.layers.SpatialDropout2D(0.1, name = tag + '_conv2_do')(conv2)
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(conv2_do)
    
    return pooling, conv2_do
    

def upsampling_block(inputs, skip_inputs, n_channels, kernel_size, pool_size, activation, tag):
    deconv = tf.keras.layers.Conv2DTranspose(n_channels, (2, 2), strides = (2, 2), padding = 'same')(inputs)
    skip_concat = tf.keras.layers.concatenate([deconv, skip_inputs])
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same')(skip_concat)
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same')(conv1)
    
    return conv2

def upsampling_block2(inputs, skip_inputs, n_channels, kernel_size, pool_size, activation, tag):
    deconv = tf.keras.layers.UpSampling2D(size = (2, 2), interpolation = 'bilinear')(inputs)
    skip_concat = tf.keras.layers.concatenate([deconv, skip_inputs])
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same')(skip_concat)
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same')(conv1)
    
    return conv2
    
def upsampling_block3(inputs, skip_inputs, n_channels, kernel_size, pool_size, activation, tag):
    deconv = pix2pix.upsample(n_channels, kernel_size)(inputs)
    skip_concat = tf.keras.layers.concatenate([deconv, skip_inputs])
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same')(skip_concat)
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, kernel_initializer = 'he_normal', padding = 'same')(conv1)
    
    return conv2
    
def upsampling_block_dilation(inputs, skip_inputs, n_channels, kernel_size, pool_size, activation, dilation_rate, tag):
    deconv = tf.keras.layers.Conv2DTranspose(n_channels, (2, 2), strides = (2, 2), padding = 'same')(inputs)
    skip_concat = tf.keras.layers.concatenate([deconv, skip_inputs])
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, dilation_rate = dilation_rate, kernel_initializer = 'he_normal', padding = 'same')(skip_concat)
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, activation = activation, dilation_rate = dilation_rate, kernel_initializer = 'he_normal', padding = 'same')(conv1)
    
    return conv2

def residual_block_skip2_bn(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv1_bn = tf.keras.layers.BatchNormalization(name = tag + '_conv1_bn')(conv1)
    conv1_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_act')(conv1_bn)
    
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1_act)
    conv2_bn = tf.keras.layers.BatchNormalization(name = tag + '_conv2_bn')(conv2)
    
    shortcut = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_shorcut')(inputs)
    shortcut_bn = tf.keras.layers.BatchNormalization(name = tag + '_shortcut_bn')(shortcut)
    
    residual = tf.keras.layers.Add(name = tag + '_add')([conv2_bn, shortcut_bn])
    residual_act = tf.keras.layers.Activation(activation, name = tag + '_add_act')(residual)
    
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(residual_act)
    
    return pooling, residual_act
    
def residual_block_skip2(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv1_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_act')(conv1)
    
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1_act)
    
    shortcut = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_shorcut')(inputs)
    
    residual = tf.keras.layers.Add(name = tag + '_add')([conv2, shortcut])
    residual_act = tf.keras.layers.Activation(activation, name = tag + '_add_act')(residual)
    
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(residual_act)
    
    return pooling, residual_act
    
def residual_block_skip2_dilation(inputs, n_channels, kernel_size, pool_size, activation, dilation_rate, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal',  dilation_rate = dilation_rate, padding = 'same', name = tag + '_conv1')(inputs)
    conv1_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_act')(conv1)
    
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', dilation_rate = dilation_rate, padding = 'same', name = tag + '_conv2')(conv1_act)
    
    shortcut = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', dilation_rate = dilation_rate, padding = 'valid', name = tag + '_shorcut')(inputs)
    
    residual = tf.keras.layers.Add(name = tag + '_add')([conv2, shortcut])
    residual_act = tf.keras.layers.Activation(activation, name = tag + '_add_act')(residual)
    
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(residual_act)
    
    return pooling, residual_act
    
def residual_block_skip2_dropout(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv1_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_act')(conv1)
    conv1_do = tf.keras.layers.SpatialDropout2D(0.1, name = tag + '_conv1_do')(conv1_act)
    
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1_do)
    
    shortcut = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_shorcut')(inputs)
    
    residual = tf.keras.layers.Add(name = tag + '_add')([conv2, shortcut])
    residual_act = tf.keras.layers.Activation(activation, name = tag + '_add_act')(residual)
    residual_do = tf.keras.layers.SpatialDropout2D(0.1, name = tag + '_residual_do')(residual_act)
    
    
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(residual_do)
    
    return pooling, residual_do

def residual_block_skip3(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1')(inputs)
    conv1_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_act')(conv1)
    
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1_act)
    conv2_act = tf.keras.layers.Activation(activation, name = tag + '_conv2_act')(conv2)
    
    conv3 = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv3')(conv2_act)
    
    shortcut = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_shorcut')(inputs)
    
    residual = tf.keras.layers.Add(name = tag + '_add')([conv3, shortcut])
    residual_act = tf.keras.layers.Activation(activation, name = tag + '_add_act')(residual)
    
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(residual_act)
    
    return pooling, residual_act
    
def residual_conv_block(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1 = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_conv1')(inputs)
    conv1_act = tf.keras.layers.Activation(activation)(conv1)
    
    conv2 = tf.keras.layers.Conv2D(n_channels, kernel_size, kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2')(conv1_act)
    conv2_act = tf.keras.layers.Activation(activation)(conv2)
    
    conv3 = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_conv3')(conv2_act)
    conv3_act = tf.keras.layers.Activation(activation)(conv3)
    
    ### kernel_size should match with block 3
    shortcut = f.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_shorcut')(inputs)
    
    residual = tf.keras.layers.Add()[con3_act, shortcut]
    residual_act = tf.keras.layers.Activation(activation)(residual)
    
    return residual_act
    
def inception_block(inputs, n_channels, kernel_size, pool_size, activation, tag):
    conv1_k3 = tf.keras.layers.Conv2D(n_channels, (3, 3), kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1_k3')(inputs)
    conv1_k3_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_k3_act')(conv1_k3)
    conv2_k3 = tf.keras.layers.Conv2D(n_channels, (3, 3), kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2_k3')(conv1_k3_act)
    conv2_k3_act = tf.keras.layers.Activation(activation, name = tag + '_conv2_k3_act')(conv2_k3)
    
    conv1_k5 = tf.keras.layers.Conv2D(n_channels, (5, 5), kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv1_k5')(inputs)
    conv1_k5_act = tf.keras.layers.Activation(activation, name = tag + '_conv1_k5_act')(conv1_k5)
    conv2_k5 = tf.keras.layers.Conv2D(n_channels, (5, 5), kernel_initializer = 'he_normal', padding = 'same', name = tag + '_conv2_k5')(conv1_k5_act)
    conv2_k5_act = tf.keras.layers.Activation(activation, name = tag + '_conv2_k5_act')(conv2_k5)
    
    concat = tf.keras.layers.concatenate([conv2_k3_act, conv2_k5_act],  name = tag + '_concat')
    concat_reduce = tf.keras.layers.Conv2D(n_channels, (1, 1), kernel_initializer = 'he_normal', padding = 'valid', name = tag + '_concat_reduce')(concat) 
    pooling = tf.keras.layers.MaxPooling2D(pool_size, name = tag + '_pool')(concat_reduce)
    
    return pooling, concat_reduce
    
    
#https://gist.github.com/ksugar/e3e0ac2e3a0beaf1b0d6e5e9cc2b84e0
