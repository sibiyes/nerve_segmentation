import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
    a = []
    b = []
    for i in range(3):
        a.append(np.random.randn(3, 3))
        b.append(np.random.randn(3, 3))
        
    a = np.array(a)
    b = np.array(b)
    
    print(a)
    print(b)
    print(np.shape(a))
    
    sigmoid_a = tf.math.sigmoid(a)
    sigmoid_b = tf.math.sigmoid(b)
    
    print(sigmoid_a)
    print(sigmoid_b)
    
    pred = tf.cast(sigmoid_a > 0.5, tf.int16)
    mask = tf.cast(sigmoid_b > 0.5, tf.int16)
    
    print(pred)
    print(mask)

    print('----------------')
    mask_pos = tf.math.reduce_sum(tf.cast(mask == 1, tf.int16))
    mask_neg = tf.math.reduce_sum(tf.cast(mask == 0, tf.int16))
    print(mask_pos, mask_neg)
    tp = tf.math.reduce_sum(pred[mask == 1])
    fp = tf.math.reduce_sum(pred[mask == 0])
    tn = mask_neg - fp
    fn = mask_pos - tp
    
    print(tp, fp, tn, fn)
    
    print(tf.math.reduce_sum(sigmoid_a, axis = (1, 2)))
    
if __name__ == '__main__':
    main()
