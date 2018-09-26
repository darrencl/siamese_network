import tensorflow as tf
from tensorflow import keras
import numpy as np

def split_dataset(x_tr,y_tr,x_t,y_t): # Split dataset into training and test sets using given integers
    # Combine datasets to be split based on integers
    X = np.concatenate((x_tr,x_t))
    Y = np.concatenate((y_tr,y_t))
    
    tr_ints , t_ints = [2,3,4,5,6,7], [0,1,8,9]
    
    # Creates a boolean mask for each set
    tr_set = [ x in tr_ints for x in Y ]
    t_set = [ x in t_ints for x in Y ]
    
    x_tr, x_t = X[tr_set], X[t_set]
    y_tr, y_t = Y[tr_set], Y[t_set]

    # Split 80% to training and 20% to test
    split = int(len(x_tr) * 0.8)    
    x_t2, x_tr = x_tr[split::], x_tr[:split:]
    y_t2, y_tr = y_tr[split::], y_tr[:split:]

    return ((x_tr, x_t, x_t2)), ((y_tr, y_t, y_t2))

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_set, y_set = split_dataset(x_train, y_train, x_test, y_test)
x_train, x_test, x_test_unknown = x_set
y_train, y_test, y_test_unknown = y_set

