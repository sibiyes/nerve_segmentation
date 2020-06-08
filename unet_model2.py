


# https://www.kaggle.com/vijaybj/basic-u-net-using-tensorflow
# https://androidkt.com/tensorflow-keras-unet-for-image-image-segmentation/
# https://towardsdatascience.com/cityscape-segmentation-with-tensorflow-2-0-b320b6605cbf
# https://readthedocs.org/projects/tf-unet/downloads/pdf/latest/
# https://towardsdatascience.com/medical-image-segmentation-part-1-unet-convolutional-networks-with-interactive-code-70f0f17f46c6

import os
import sys

import tensorflow as tf
#from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np
import matplotlib.pyplot as plt
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

from load_data_utils import load_data
from loss import weighted_binary_crossentropy
from blocks import downsampling_block, downsampling_block_bn, upsampling_block, upsampling_block2, upsampling_block3, downsampling_block_dropout
from blocks import downsampling_block_dilation, upsampling_block_dilation

def model1(model_tag, size, activation, weight):
    
    IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS = size
    
    ### down 1
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    block1_out, block1_skip_out = downsampling_block(inputs, 16, (3, 3), (2, 2), activation, 'down1')
    
    ### down 2
    block2_out, block2_skip_out = downsampling_block(block1_out, 32, (3, 3), (2, 2), activation, 'down2')
    
    ### down 3
    block3_out, block3_skip_out = downsampling_block(block2_out, 64, (3, 3), (2, 2), activation, 'down3')
    
    ### down 4
    block4_out, block4_skip_out = downsampling_block(block3_out, 128, (3, 3), (2, 2), activation, 'down4')
    
    ### down5
    block5_out, block5_skip_out = downsampling_block(block4_out, 256, (3, 3), (2, 2), activation, 'down5')
    
    ### connector
    block6a = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block5_out)
    block6b = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block6a)
    
    ### up 1
    block7_out = upsampling_block(block6b, block5_skip_out, 256, (3, 3), (2, 2), activation, 'up1')
    
    ### up 2
    block8_out = upsampling_block(block7_out, block4_skip_out, 128, (3, 3), (2, 2), activation, 'up2')
    
    ### up 3
    block9_out = upsampling_block(block8_out, block3_skip_out, 64, (3, 3), (2, 2), activation, 'up3')
    
    ### up 4
    block10_out = upsampling_block(block9_out, block2_skip_out, 32, (3, 3), (2, 2), activation, 'up4')
    
    ### up 5
    block11_out = upsampling_block(block10_out, block1_skip_out, 16, (3, 3), (2, 2), activation, 'up5')
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(block11_out)
    
    
    
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = optimizer, loss = weighted_binary_crossentropy(pos_weight=weight), metrics = ['accuracy'])
    model.summary()
    
    return model
    
def model1a(model_tag, size, activation, weight):
    
    IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS = size
    
    ### down 1
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    block1_out, block1_skip_out = downsampling_block(inputs, 16, (3, 3), (2, 2), activation, 'down1')
    block2_out, block2_skip_out = downsampling_block(block1_out, 32, (3, 3), (2, 2), activation, 'down2')
    block3_out, block3_skip_out = downsampling_block(block2_out, 64, (3, 3), (2, 2), activation, 'down3')
    block4_out, block4_skip_out = downsampling_block(block3_out, 128, (3, 3), (2, 2), activation, 'down4')
    block5_out, block5_skip_out = downsampling_block(block4_out, 256, (3, 3), (2, 2), activation, 'down5')
    
    ### connector
    block6a = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block5_out)
    block6b = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block6a)
    
    ### up 1
    block7_out = upsampling_block2(block6b, block5_skip_out, 256, (3, 3), (2, 2), activation, 'up1')
    block8_out = upsampling_block2(block7_out, block4_skip_out, 128, (3, 3), (2, 2), activation, 'up2')
    block9_out = upsampling_block2(block8_out, block3_skip_out, 64, (3, 3), (2, 2), activation, 'up3')
    block10_out = upsampling_block2(block9_out, block2_skip_out, 32, (3, 3), (2, 2), activation, 'up4')
    block11_out = upsampling_block2(block10_out, block1_skip_out, 16, (3, 3), (2, 2), activation, 'up5')
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(block11_out)
    
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = optimizer, loss = weighted_binary_crossentropy(pos_weight=weight), metrics = ['accuracy'])
    model.summary()
    
    return model
    
def model1b(model_tag, size, activation, weight):
    
    IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS = size
    
    ### down 1
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    block1_out, block1_skip_out = downsampling_block(inputs, 16, (3, 3), (2, 2), activation, 'down1')
    block2_out, block2_skip_out = downsampling_block(block1_out, 32, (3, 3), (2, 2), activation, 'down2')
    block3_out, block3_skip_out = downsampling_block(block2_out, 64, (3, 3), (2, 2), activation, 'down3')
    block4_out, block4_skip_out = downsampling_block(block3_out, 128, (3, 3), (2, 2), activation, 'down4')
    block5_out, block5_skip_out = downsampling_block(block4_out, 256, (3, 3), (2, 2), activation, 'down5')
    
    ### connector
    block6a = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block5_out)
    block6b = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block6a)
    
    ### up 1
    block7_out = upsampling_block3(block6b, block5_skip_out, 256, 3, (2, 2), activation, 'up1')
    block8_out = upsampling_block3(block7_out, block4_skip_out, 128, 3, (2, 2), activation, 'up2')
    block9_out = upsampling_block3(block8_out, block3_skip_out, 64, 3, (2, 2), activation, 'up3')
    block10_out = upsampling_block3(block9_out, block2_skip_out, 64, 3, (2, 2), activation, 'up4')
    block11_out = upsampling_block3(block10_out, block1_skip_out, 64, 3, (2, 2), activation, 'up5')
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(block11_out)
    
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = optimizer, loss = weighted_binary_crossentropy(pos_weight=weight), metrics = ['accuracy'])
    model.summary()
    
    return model
    
def model2(model_tag, size, activation, weight):
    
    IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS = size
    
    
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    ### down blocks
    block1_out, block1_skip_out = downsampling_block_bn(inputs, 16, (3, 3), (2, 2), activation, 'down1')
    block2_out, block2_skip_out = downsampling_block_bn(block1_out, 32, (3, 3), (2, 2), activation, 'down2')
    block3_out, block3_skip_out = downsampling_block_bn(block2_out, 64, (3, 3), (2, 2), activation, 'down3')
    block4_out, block4_skip_out = downsampling_block_bn(block3_out, 128, (3, 3), (2, 2), activation, 'down4')
    block5_out, block5_skip_out = downsampling_block_bn(block4_out, 256, (3, 3), (2, 2), activation, 'down5')
    
    ### connector
    block6a = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer = 'he_normal', padding = 'same', name =  'conn1_conv')(block5_out)
    block6a_bn = tf.keras.layers.BatchNormalization(name = 'conn1_bn')(block6a)
    block6a_act = tf.keras.layers.Activation(activation, name =  'conn1_act')(block6a_bn)
    
    block6b = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer = 'he_normal', padding = 'same', name =  'conn2_conv')(block6a_act)
    block6b_bn = tf.keras.layers.BatchNormalization(name = 'conn2_bn')(block6b)
    block6b_act = tf.keras.layers.Activation(activation, name =  'conn2_act')(block6b_bn)
    
    ### up blocks
    block7_out = upsampling_block(block6b_act, block5_skip_out, 256, (3, 3), (2, 2), activation, 'up1')
    block8_out = upsampling_block(block7_out, block4_skip_out, 128, (3, 3), (2, 2), activation, 'up2')
    block9_out = upsampling_block(block8_out, block3_skip_out, 64, (3, 3), (2, 2), activation, 'up3')
    block10_out = upsampling_block(block9_out, block2_skip_out, 32, (3, 3), (2, 2), activation, 'up4')
    block11_out = upsampling_block(block10_out, block1_skip_out, 16, (3, 3), (2, 2), activation, 'up5')
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(block11_out)

    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = optimizer, loss = weighted_binary_crossentropy(pos_weight=weight), metrics = ['accuracy'])
    model.summary()
    
    return model
    
def model3(model_tag, size, activation, weight):
    
    IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS = size
    
    
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    ### down blocks
    block1_out, block1_skip_out = downsampling_block_dropout(inputs, 16, (3, 3), (2, 2), activation, 'down1')
    block2_out, block2_skip_out = downsampling_block_dropout(block1_out, 32, (3, 3), (2, 2), activation, 'down2')
    block3_out, block3_skip_out = downsampling_block_dropout(block2_out, 64, (3, 3), (2, 2), activation, 'down3')
    block4_out, block4_skip_out = downsampling_block_dropout(block3_out, 128, (3, 3), (2, 2), activation, 'down4')
    block5_out, block5_skip_out = downsampling_block_dropout(block4_out, 256, (3, 3), (2, 2), activation, 'down5')
    
    ### connector
    block6a = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer = 'he_normal', padding = 'same', name =  'conn1_conv')(block5_out)
    block6a_bn = tf.keras.layers.BatchNormalization(name = 'conn1_bn')(block6a)
    block6a_act = tf.keras.layers.Activation(activation, name =  'conn1_act')(block6a_bn)
    
    block6b = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer = 'he_normal', padding = 'same', name =  'conn2_conv')(block6a_act)
    block6b_bn = tf.keras.layers.BatchNormalization(name = 'conn2_bn')(block6b)
    block6b_act = tf.keras.layers.Activation(activation, name =  'conn2_act')(block6b_bn)
    
    ### up blocks
    block7_out = upsampling_block(block6b_act, block5_skip_out, 256, (3, 3), (2, 2), activation, 'up1')
    block8_out = upsampling_block(block7_out, block4_skip_out, 128, (3, 3), (2, 2), activation, 'up2')
    block9_out = upsampling_block(block8_out, block3_skip_out, 64, (3, 3), (2, 2), activation, 'up3')
    block10_out = upsampling_block(block9_out, block2_skip_out, 32, (3, 3), (2, 2), activation, 'up4')
    block11_out = upsampling_block(block10_out, block1_skip_out, 16, (3, 3), (2, 2), activation, 'up5')
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(block11_out)

    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = optimizer, loss = weighted_binary_crossentropy(pos_weight=weight), metrics = ['accuracy'])
    model.summary()
    
    return model
    
def model4(model_tag, size, activation, weight):
    
    IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS = size
    
    ### down 1
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    block1_out, block1_skip_out = downsampling_block_dilation(inputs, 16, (3, 3), (2, 2), activation, (2, 2), 'down1')
    block2_out, block2_skip_out = downsampling_block_dilation(block1_out, 32, (3, 3), (2, 2), activation, (2, 2), 'down2')
    block3_out, block3_skip_out = downsampling_block_dilation(block2_out, 64, (3, 3), (2, 2), activation, (2, 2), 'down3')
    block4_out, block4_skip_out = downsampling_block(block3_out, 128, (3, 3), (2, 2), activation, 'down4')
    block5_out, block5_skip_out = downsampling_block(block4_out, 256, (3, 3), (2, 2), activation, 'down5')
    
    ### connector
    block6a = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block5_out)
    block6b = tf.keras.layers.Conv2D(512, (3, 3), activation = activation, kernel_initializer = 'he_normal', padding = 'same')(block6a)
    
    ### up 1
    block7_out = upsampling_block(block6b, block5_skip_out, 256, (3, 3), (2, 2), activation, 'up1')
    block8_out = upsampling_block(block7_out, block4_skip_out, 128, (3, 3), (2, 2), activation, 'up2')
    block9_out = upsampling_block_dilation(block8_out, block3_skip_out, 64, (3, 3), (2, 2), activation, (2, 2), 'up3')
    block10_out = upsampling_block_dilation(block9_out, block2_skip_out, 32, (3, 3), (2, 2), activation, (2, 2), 'up4')
    block11_out = upsampling_block_dilation(block10_out, block1_skip_out, 16, (3, 3), (2, 2), activation, (2, 2), 'up5')
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(block11_out)
    
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = optimizer, loss = weighted_binary_crossentropy(pos_weight=weight), metrics = ['accuracy'])
    model.summary()
    
    return model
    
    
def run_model():
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_data_sample'
    model_dir = base_dir + '/models'
    
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    NUM_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    
    train_data = load_data(data_dir, [IMG_WIDTH, IMG_HEIGHT])
    print(train_data)
    
    #N = 5635
    N = 500
    
    tag = 'unet_model_b5_relu_dil_w10'
    model_dir += '/{0}'.format(tag)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # for image, mask in train_data.take(1):
    #     sample_image, sample_mask = image, mask
    
    
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
    
    model = model4(tag, (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS), 'relu', 10)
    
    model_save_callback = tf.keras.callbacks.ModelCheckpoint(
                                    model_dir + '/model_epoch_{epoch:04d}.h5',
                                    save_weights_only = False,
                                    period = 25)
    
    model_history = model.fit(train_dataset, epochs = EPOCHS,
                              steps_per_epoch = STEPS_PER_EPOCH,
                              validation_steps = VALIDATION_STEPS,
                              validation_data = train_dataset,
                              callbacks = [model_save_callback])
                              
    sigmoid = lambda x: 1/(1+np.exp(-1*x))
    
    
    
    model.save(model_dir + '/model_final.h5')
    
    
def main():
    run_model()
    
if __name__ == '__main__':
    main()
