import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

from model_simple import display
from load_data_utils import load_data
from loss import weighted_binary_crossentropy, dice_coefficient, tversky_loss

def custom_metric(y_true, y_pred):
    return 0.5

        
def predict(tag, data_tag, n):
    # tag = 'unet_model1_elu2'
    # data_tag = 'test'
    # n = 500
    
    
    model_dir = base_dir + '/models'
    
    #tag = 'unet_model_b5_relu'
    model_dir += '/{0}'.format(tag)
    
    
    if (data_tag == 'train'):
        data_dir = base_dir + '/ultrasound-nerve-segmentation/train_data_sample'
    else:
        data_dir = base_dir + '/ultrasound-nerve-segmentation/test_data_sample'
        
    predictions_dir = base_dir + '/predictions_epochs/{0}/{1}'.format(tag, data_tag)
    prediction_results_dir = base_dir + '/prediction_results_epochs/{0}/{1}'.format(tag, data_tag)
    prediction_plots_dir = base_dir + '/plots/predictions_epochs/{0}/{1}'.format(tag, data_tag)
    
    if (not os.path.exists(predictions_dir)):
        os.makedirs(predictions_dir)
        
    if (not os.path.exists(prediction_results_dir)):
        os.makedirs(prediction_results_dir)
    
    if (not os.path.exists(prediction_plots_dir)):
        os.makedirs(prediction_plots_dir)
        
    loss = weighted_binary_crossentropy(pos_weight=10)
    dice_loss = dice_coefficient()
    tvk_loss = tversky_loss(0.9)
    #model = tf.keras.models.load_model(model_dir + '/{0}.h5'.format(tag))
    #model = tf.keras.models.load_model(model_dir + '/model_final.h5', custom_objects = {'_weighted_binary_crossentropy': loss, '_dice_coefficient': dice_loss, '_tversky_loss': tvk_loss})
    
    # IMG_HEIGHT = 128
    # IMG_WIDTH = 128
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    
    if ('b5' in tag ):
        if (IMG_HEIGHT != 256):
            print('SIZE MISMATCH')
            sys.exit(0)
    
    NUM_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    
    data = load_data(data_dir, [IMG_WIDTH, IMG_HEIGHT])
    
    sigmoid = lambda x: 1/(1+np.exp(-1*x))
    
    
    epochs = np.arange(275, 501, 25)
    #probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prob = 0.5
    
    for epoch in epochs:
        model = tf.keras.models.load_model(model_dir + '/model_epoch_{epoch:04d}.h5'.format(epoch = epoch), custom_objects = {'_weighted_binary_crossentropy': loss, '_dice_coefficient': dice_loss, '_tversky_loss': tvk_loss})
        
        fp_out = open(prediction_results_dir + '/results_img_{0}.csv'.format(epoch), 'w')
        fp_out.write('img,pos_pixels,neg_pixels,tp,fp,tn,fn' + '\n')
        
        y_pos_total = 0
        y_neg_total = 0
        tp_total = 0
        fp_total = 0
        tn_total = 0
        fn_total = 0
        dice_coeff_sum = 0
        iou_sum = 0
        
        i = 0
        dice_cnt = 0
        iou_cnt = 0
        cnt = 0
        for image, mask in data.take(n):
            pred_mask = model.predict(image[tf.newaxis, ...])
            pred_mask = pred_mask[0, :, :, :]
            pred_mask = sigmoid(pred_mask)
            print(np.mean(pred_mask))
            
            fp = open(predictions_dir + '/true_mask_{0}.pkl'.format(i), 'wb')
            pickle.dump(mask, fp)
            fp.close()
            fp = open(predictions_dir + '/pred_mask_{0}.pkl'.format(i), 'wb')
            pickle.dump(pred_mask, fp)
            fp.close()
            
            # print(np.round(pred_mask, 2))
            # print(mask)
            
            #0.500005, 0.500001, 0.50001
            #pred_mask_label = (pred_mask > 0.6).astype(float)
            #pred_mask_label = (pred_mask > 0.015).astype(float)
            #pred_mask_label = (pred_mask > 0.5001).astype(float)
            
            pred_mask_label = (pred_mask > prob).astype(float)
            
            true_mask_pos_index = (mask == 1.0)
            true_mask_neg_index = (mask != 1.0)
            # print(true_mask_pos_index)
            # print(true_mask_neg_index)
            y_pos = np.sum(true_mask_pos_index)
            y_neg = np.sum(true_mask_neg_index)
            
            tp = np.sum(pred_mask_label[true_mask_pos_index])
            fp = np.sum(pred_mask_label[true_mask_neg_index])
            tn = y_neg - fp
            fn = y_pos - tp
            dice_coeff = round((2*tp)/(2*tp + fp + fn), 6) if tp + fp + fn != 0 else 1.0
            iou = round(tp/(tp + fp + fn), 6) if tp + fp + fn != 0 else 1.0
            
            print(y_pos, y_neg)
            print(tp, fp, tn, fn)
            print(dice_coeff, iou)
            
            fp_out.write(','.join(['img{0}'.format(i), str(y_pos), str(y_neg), str(tp), str(fp), str(tn), str(fn), str(dice_coeff), str(iou)]) + '\n')
            
            y_pos_total += y_pos
            y_neg_total += y_neg
            tp_total += tp
            fp_total += fp
            tn_total += tn
            fn_total += fn
            dice_coeff_sum += dice_coeff
            iou_sum += iou
            
            # if (not np.isnan(dice_coeff)):
            #     dice_coeff_sum += dice_coeff
            #     dice_cnt += 1
            # if (not np.isnan(iou)):
            #     iou_sum += iou
            #     iou_cnt += 1
                
            cnt += 1
            
            conf_matrix = np.array([[tn, fp], [fn, tp]]).astype(int)
            print(conf_matrix)
            
            #display([image, mask, pred_mask_label], prediction_plots_dir + '/img_{0}.png'.format(i))
            #display([image, mask, pred_mask_label])
        
        
        #sys.exit(0)
        
            i += 1
        
        dice_coeff_avg = round(dice_coeff_sum/cnt, 6)
        iou_avg = round(iou_sum/cnt, 6)
        
        dice_coeff_agg = round((2*tp_total)/(2*tp_total + fp_total + fn_total), 6)
        iou_agg = round(tp_total/(tp_total + fp_total + fn_total), 6)
        
        output_line = 'total,' + ','.join([str(x) for x in [y_pos_total, y_neg_total, tp_total, fp_total, tn_total, fn_total, dice_coeff_avg, iou_avg, dice_coeff_agg, iou_agg]])
        
        fp_out.write(output_line + '\n')
    
        fp_out.close()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_tag', required = True)
    parser.add_argument('--data_tag', required = True)
    parser.add_argument('--n', required = True)
    
    input_args = parser.parse_args()
    print(input_args)
    
    model_tag = input_args.model_tag
    data_tag = input_args.data_tag
    n = int(input_args.n)
    
    predict(model_tag, data_tag, n)
    
if __name__ == '__main__':
    main()
