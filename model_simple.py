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
from loss import weighted_binary_crossentropy

def display(display_list, save_path = None):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    
    if (save_path is None):
        plt.show()
    else:
        fig = plt.gcf()
        fig.set_size_inches((16, 11))
        fig.savefig(save_path)
        
        plt.clf()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    return pred_mask[0]

def show_predictions(model, dataset = None, num= 1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image[tf.newaxis, ...])
            #display([image[0], mask[0], create_mask(pred_mask)])
            display([image, mask, create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])
        
    # for image, mask in model.training_data.take(1):
    #     pred_mask = model.predict(image[tf.newaxis, ...])
    #     #display([image[0], mask[0], create_mask(pred_mask)])
    #     display([image, mask, create_mask(pred_mask)])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #clear_output(wait=True)
        show_predictions(self.model)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def main():
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train_sample'
    model_dir = base_dir + '/models'
    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    NUM_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    
    train_data = load_data(data_dir, [IMG_WIDTH, IMG_HEIGHT])
    
    #N = 5635
    N = 500
    tag = 'model_simple2_w25'
    
    # for image, mask in train_data.take(1):
    #     sample_image, sample_mask = image, mask
    
    base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top = False)
    
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    
    layers = [base_model.get_layer(name).output for name in layer_names]
    
    print(layers)
    
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs = base_model.input, outputs = layers)
    print(down_stack)
    
    down_stack.trainable = False
    
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
                  
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS, 3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)
    #outputs = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(x)
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(x)
    
    # METRICS = [
    #     'accuracy',
    #     tf.keras.metrics.Precision(),
    #     tf.keras.metrics.Recall()
    # ]
    
    def custom_metric(y_true, y_pred):
        return 0.5
    
    # METRICS = [
    #     'accuracy',
    #     custom_metric
    # ]
    
    METRICS = [
        'accuracy'
    ]
    
    # model = tf.keras.Model(inputs=inputs, outputs = x)
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               metrics=METRICS)
    
    model_save_callback = tf.keras.callbacks.ModelCheckpoint(
                                    model_dir + '/model_epoch_{epoch:04d}.h5',
                                    save_weights_only = True,
                                    period = 25
                                )
                  
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    #model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=METRICS)
    model.compile(optimizer = 'adam', loss = weighted_binary_crossentropy(pos_weight=25), metrics = ['accuracy'])
                  
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
                  
                  
    print(model)
                  
    #tf.keras.utils.plot_model(model, show_shapes=True)
    
    #show_predictions(model, train_data, 1)
    
    ########################################
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
    
    # model_history = model.fit(train_data, epochs=EPOCHS,
    #                           steps_per_epoch=STEPS_PER_EPOCH,
    #                           validation_steps=VALIDATION_STEPS,
    #                           validation_data=train_data,
    #                           callbacks=[DisplayCallback()])
    
    class_weight = {0: 2.0, 1: 98.0}
                              
    # model_history = model.fit(train_dataset, epochs = EPOCHS,
    #                           steps_per_epoch = STEPS_PER_EPOCH,
    #                           validation_steps = VALIDATION_STEPS,
    #                           validation_data = train_dataset,
    #                           class_weight = class_weight)
                              
    model_history = model.fit(train_dataset, epochs = EPOCHS,
                              steps_per_epoch = STEPS_PER_EPOCH,
                              validation_steps = VALIDATION_STEPS,
                              validation_data = train_dataset,
                              callbacks = [model_save_callback]
                              )
                              
                            
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(EPOCHS)

    # plt.figure()
    # plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss Value')
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()

    #show_predictions(model, train_data, 1)
    sigmoid = lambda x: 1/(1+np.exp(-1*x))
    
    
    
    model.save(model_dir + '/{0}.h5'.format(tag))
    
    
    
if __name__ == '__main__':
    main()
