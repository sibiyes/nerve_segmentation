


# https://www.kaggle.com/vijaybj/basic-u-net-using-tensorflow
# https://androidkt.com/tensorflow-keras-unet-for-image-image-segmentation/
# https://towardsdatascience.com/cityscape-segmentation-with-tensorflow-2-0-b320b6605cbf
# https://readthedocs.org/projects/tf-unet/downloads/pdf/latest/
# https://towardsdatascience.com/medical-image-segmentation-part-1-unet-convolutional-networks-with-interactive-code-70f0f17f46c6

import os
import sys

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np
import matplotlib.pyplot as plt
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

from load_data_utils import load_data
from blocks import downsampling_block, upsampling_block
from loss import weighted_binary_crossentropy, dice_coefficient, tversky_loss


def model1():
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_data_sample'
    model_dir = base_dir + '/models'
    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    NUM_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    
    train_data = load_data(data_dir, [IMG_WIDTH, IMG_HEIGHT])
    print(train_data)
    
    #N = 5635
    N = 500
    
    tag = 'unet_model1_rrelu_dice'
    model_dir += '/{0}'.format(tag)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # for image, mask in train_data.take(1):
    #     sample_image, sample_mask = image, mask
    
    
    ### down 1
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    
    pooling1, conv1b = downsampling_block(inputs, 16, (3, 3), (2, 2), 'relu', 'down1')
    
    ### down 2
    pooling2, conv2b = downsampling_block(pooling1, 32, (3, 3), (2, 2), 'relu', 'down2')
    
    ### down 3
    pooling3, conv3b = downsampling_block(pooling2, 64, (3, 3), (2, 2), 'relu', 'down3')
    
    ### down 4
    pooling4, conv4b = downsampling_block(pooling3, 128, (3, 3), (2, 2), 'relu', 'down4')
    
    ### down 5
    conv5a = tf.keras.layers.Conv2D(256, (3, 3), activation = tf.keras.activations.relu, kernel_initializer = 'he_normal', padding = 'same')(pooling4)
    conv5b = tf.keras.layers.Conv2D(256, (3, 3), activation = tf.keras.activations.relu, kernel_initializer = 'he_normal', padding = 'same')(conv5a)
    
    ### up 1
    conv6b = upsampling_block(conv5b, conv4b, 128, (3, 3), (2, 2), 'relu', 'up1')
    
    ### up 2
    conv7b = upsampling_block(conv6b, conv3b, 64, (3, 3), (2, 2), 'relu', 'up2')
    
    ### up 3    
    conv8b = upsampling_block(conv7b, conv2b, 32, (3, 3), (2, 2), 'relu', 'up3')
    
    ### up 4    
    conv9b = upsampling_block(conv8b, conv1b, 16, (3, 3), (2, 2), 'relu', 'up4')
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(conv9b)
    #outputs = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(conv9b)
    
    model_save_callback = tf.keras.callbacks.ModelCheckpoint(
                                    model_dir + '/model_epoch_{epoch:04d}.h5',
                                    save_weights_only = True,
                                    period = 25
                                )
    
    #model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', from_logits = True, metrics = ['accuracy'])
    #model.compile(optimizer = 'adam', loss = weighted_binary_crossentropy(pos_weight=10), from_logits = True, metrics = ['accuracy'])
    #model.compile(optimizer = 'adam', loss = weighted_binary_crossentropy(pos_weight=10), metrics = ['accuracy'])
    model.compile(optimizer = 'adam', loss = dice_coefficient(), metrics = ['accuracy'])
    #model.compile(optimizer = 'adam', loss = tversky_loss(0.1), metrics = ['accuracy'])
    model.summary()
    
    #sys.exit(0)
    
    # model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=20,
    #                 callbacks=callbacks)
    
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = N // BATCH_SIZE
    
    train_dataset = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #test_dataset = test.batch(BATCH_SIZE)
    
    EPOCHS = 250
    VAL_SUBSPLITS = 5
    #VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
    VALIDATION_STEPS = N//BATCH_SIZE//VAL_SUBSPLITS
    
    
    
    model_history = model.fit(train_dataset, epochs = EPOCHS,
                              steps_per_epoch = STEPS_PER_EPOCH,
                              validation_steps = VALIDATION_STEPS,
                              validation_data = train_dataset,
                              callbacks = [model_save_callback]
                              )
                              
    sigmoid = lambda x: 1/(1+np.exp(-1*x))
    
    
    model.save(model_dir + '/model_final.h5')
    
    
    
def main():
    model1()
    
if __name__ == '__main__':
    main()
