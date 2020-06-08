#https://gist.github.com/ksugar/e3e0ac2e3a0beaf1b0d6e5e9cc2b84e0

import tensorflow as tf
from keras import backend as K

def weighted_binary_crossentropy(pos_weight=1):
    def _to_tensor(x, dtype):
        return tf.convert_to_tensor(x, dtype=dtype)
  
  
    def _calculate_weighted_binary_crossentropy(target, output):
        return tf.nn.weighted_cross_entropy_with_logits(target, output, pos_weight)


    def _weighted_binary_crossentropy(y_true, y_pred):
        return K.mean(_calculate_weighted_binary_crossentropy(y_true, y_pred), axis=-1)
    
    return _weighted_binary_crossentropy
    
def dice_coefficient2():
    
    def _dice_coefficient(y_true, y_pred):
        y_pred_sigmod = tf.math.sigmoid(y_pred)
        
        numerator = 2 * tf.reduce_sum(y_true * y_pred_sigmod, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred_sigmod, axis=(1,2,3))

        dice =  1 - numerator/denominator
        
        return dice
        
    return _dice_coefficient
    
    
def dice_coefficient3():
    
    def _dice_coefficient(y_true, y_pred):
        y_pred_sigmod = tf.math.sigmoid(y_pred)
        
        numerator = 2 * tf.reduce_sum(y_true * y_pred_sigmod, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred_sigmod, axis=-1)

        dice =  1 - (numerator + 1)/(denominator + 1)
        
        return dice
        
    return _dice_coefficient
    
def dice_coefficient(epsilon = 1e-6):
    
    def _dice_coefficient(y_true, y_pred):
        y_pred_sigmod = tf.math.sigmoid(y_pred)
        
        numerator = 2 * tf.reduce_sum(y_true * y_pred_sigmod, axis=-1)
        denominator = tf.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred_sigmod), axis=-1)

        dice =  1 - (numerator)/(denominator + epsilon)
        
        return dice
        
    return _dice_coefficient
    
def tversky_loss(beta):
    def _tversky_loss(y_true, y_pred):
        y_pred_sigmod = tf.math.sigmoid(y_pred)
        
        numerator = tf.reduce_sum(y_true * y_pred_sigmod, axis=-1)
        denominator = y_true * y_pred_sigmod + beta * (1 - y_true) * y_pred_sigmod + (1 - beta) * y_true * (1 - y_pred_sigmod)

        return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

    return _tversky_loss
    
# smooth = 1.
# 
# 
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# 
# 
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)


        
