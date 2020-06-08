import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

def plots(data):
    ax = sns.lineplot(x = 'epoch', y = 'precision', hue = 'model', style = 'data', markers = True, data = data)
    ax.set_title('precision by epochs', size = 36)
    ax.set_xlabel('epochs', fontsize=26)
    ax.set_ylabel('precision', fontsize=26)
    plt.legend(fontsize='xx-large', title_fontsize='60')
    
    plt.gcf().set_size_inches(20, 10)
    plt.savefig(base_dir + '/precision_epochs.png')
    plt.clf()
    
    #plt.show()

def main():
    #models = ['unet_model1_relu_w10', 'unet_model1_elu_w10', 'unet_model1_relu_w25', 'unet_model1_elu_w25']
    # models = [
    #     'unet_model_b5_relu_do_w10',
    #     'unet_model_b5_relu_do_w25',
    #     'resnet_model_b5_relu_do_w10',
    #     'resnet_model_b5_relu_do_w25'
    # ]
    
    models = [
        'resnet_model_b5_relu_w10_n750',
        'resnet_model_b5_relu_w10_n1000',
        'unet_model_b5_relu_w10_n750',
        'unet_model_b5_relu_w10_n1000'
    ]
    
    agg_results = []
    for model_tag in models:
        prediction_results_dir = base_dir + '/prediction_results_epochs'
        
        #tag = 'unet_model_b5_relu'
        prediction_results_dir += '/{0}'.format(model_tag)
        print(prediction_results_dir)
    
        for data in ['train', 'test']:
            pred_files = os.listdir(prediction_results_dir + '/' + data)
            for pred_file in pred_files:
                fp = open(prediction_results_dir + '/' + data + '/' + pred_file, 'r')
                lines = fp.readlines()
                fp.close()
                
                file_no_ext = pred_file.replace('.csv', '')
                epoch = file_no_ext.split('_')[-1]
                
                agg_results.append([model_tag, data, epoch] + [float(x) for x in lines[-1].strip().split(',')[1:]])
        
    agg_results_df = pd.DataFrame(agg_results, columns = ['model', 'data', 'epoch', 'y_pos_total', 'y_neg_total', 'tp_total', 'fp_total', 'tn_total', 'fn_total', 'dice_coeff_avg', 'iou_avg', 'dice_coeff_agg', 'iou_agg'])
    agg_results_df = agg_results_df.astype({
                    'epoch': int,
                    'y_pos_total': int, 
                    'y_neg_total': int, 
                    'tp_total': int,
                    'fp_total': int,
                    'tn_total': int,
                    'fn_total': int, 
                    'dice_coeff_avg': float, 
                    'iou_avg': float, 
                    'dice_coeff_agg': float,
                    'iou_agg': float}
        )
    
    
    
    agg_results_df['precision'] =  agg_results_df['tp_total']/(agg_results_df['tp_total'] + agg_results_df['fp_total'])
    agg_results_df['recall'] =  agg_results_df['tp_total']/(agg_results_df['tp_total'] + agg_results_df['fn_total'])
    agg_results_df['f_score'] = 2*((agg_results_df['precision']*agg_results_df['recall'])/agg_results_df['precision'] + agg_results_df['recall'])
    
    print(agg_results_df)
    plots(agg_results_df)
    sys.exit(0)
    
    agg_results_df = agg_results_df.sort_values(by = ['model', 'data', 'prob'])
    
    
    agg_results_filter = agg_results_df[agg_results_df['prob'].astype(float) == 0.5]
    agg_results_filter = agg_results_filter.sort_values(by = ['model', 'data'], ascending = [True, False])
    print(agg_results_filter)
    agg_results_filter.to_csv(base_dir + '/results_filter.csv', index = None)
    
    
    
if __name__ == '__main__':
    main()
    
    
