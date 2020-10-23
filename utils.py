import numpy as np
import pickle
import os
import tensorflow as tf

def load_batch(path,label_key='labels'):
    with open(path,'rb') as f:
        d = pickle.load(f)
        d_dcoded = {}
        for k,v in d.items():
            d_dcoded[k.decode('utf8')] = v
        d = d_dcoded
    data = d['data']
    labels = d[label_key]
    data = data.reshape(data.shape[0],3,32,32)

    return data,labels
def load_dat():
    path = 'cifar-10-batches-py'
    train_samples = 50000


    x_train = np.empty((train_samples,3,32,32),dtype='unit8')
    y_train = np.empty((train_samples,),dtype='unit8')

    test_path = os.path.join(path,'test_batch')
    x_test,y_test = load_batch(test_path)

