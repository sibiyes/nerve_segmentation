import os
import sys

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

def main():
    
    models = ['model_simple2', 'unet_model1_elu2', 'unet_model1_relu2']
    
    for model_tag in models:
        for data_tag in ('train', 'test'):
            prediction_results_dir = base_dir + '/prediction_results/{0}/{1}'.format(model_tag, data_tag)
            
            probs = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
            for prob in probs:
                fp_in = open(prediction_results_dir + '/results_img_{0}.csv'.format(prob), 'rb')
                results = pd.read_csv(fp_in)
                results_agg = results.values[-1][1:]
                #print(results_agg)
                precision = results_agg[2]/(results_agg[2] + results_agg[3])
                recall = results_agg[2]/(results_agg[2] + results_agg[5])
                fp_in.close()
                
                print([model_tag, data_tag, prob] + [int(x) for x in results_agg] + [str(round(precision, 3)), str(round(recall, 3))])
    
if __name__ == '__main__':
    main()
