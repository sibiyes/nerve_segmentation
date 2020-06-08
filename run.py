
   
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
    
    tag = 'unet_model_b5_relu_us_w10'
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
    
    model = model1a(tag, (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, OUTPUT_CHANNELS), 'relu', 10)
    
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
