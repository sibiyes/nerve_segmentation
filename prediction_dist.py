import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

def main():
    
    N = 500
    tag = 'model_simple2'
    data_tag = 'train'
    
    predictions_dir = base_dir + '/predictions/{0}/{1}'.format(tag, data_tag)
    
    i = 0
    
    interval_lower = np.arange(-0.01, 1.0, 0.01)
    interval_upper = np.arange(0.0, 1.01, 0.01)
    
    intervals = list(zip(interval_lower, interval_upper))
    
    vals_all = []
    for i in range(N):
        fp = open(predictions_dir + '/true_mask_{0}.pkl'.format(i), 'rb')
        true_mask = pickle.load(fp)
        fp.close()
        
        fp = open(predictions_dir + '/pred_mask_{0}.pkl'.format(i), 'rb')
        pred_mask = pickle.load(fp)
        fp.close()
        
        # print(pred_mask)
        # print(np.shape(pred_mask))
        
        # intervals = [
        #     (0.0, 0.1),
        #     (0.1, 0.2),
        #     (0.2, 0.3),
        #     (0.3, 0.4),
        #     (0.4, 0.5),
        #     (0.5, 0.6),
        #     (0.6, 0.7),
        #     (0.7, 0.8),
        #     (0.8, 0.9),
        #     (0.9, 1.01)
        # ]
        
        
        
        vals = []
        for interval in intervals:
            cnt = np.sum((pred_mask > interval[0]) & (pred_mask <= interval[1]))
            vals.append(cnt)
            #print(interval)
            
        #print(vals)
        
        vals_all.append(vals)
        
    vals_all = np.array(vals_all)
    
    #print(vals_all)
    vals_all_agg  = np.sum(vals_all, axis = 0)
    vals_all_agg_perc = vals_all_agg/np.sum(vals_all_agg)
    
    for x in zip(intervals, np.round(vals_all_agg_perc, 4).astype(str)):
        print(x)
        
    #sys.exit(0)
    
if __name__ == '__main__':
    main()
