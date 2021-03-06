# # -*- coding: UTF-8 -*-


# 保证脚本与Python3兼容
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
import pandas as pd
import linear_reg as lr

def read_data(train_file_path,test_file_path,sep=','):
    data_path = os.path.join(os.path.curdir,'data')
    train_file_path=os.path.join(data_path,train_file_path)
    test_file_path=os.path.join(data_path,test_file_path)
    train_df=pd.read_csv(train_file_path,sep=sep,header=None)
    test_df=pd.read_csv(test_file_path,sep=sep,header=None)
    return (train_df,test_df)

def split_train_test(x_vals,y_vals,rate=0.8):
    """
    split the data into test and train
    """
    train_indices = np.random.choice(len(x_vals),int(round(len(x_vals)*rate)), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]
    return (x_vals_train,y_vals_train,x_vals_test,y_vals_test)


def gradientDescent(X,Y,X_test,Y_test,model,learningRate=0.01,maxIter=10000,tol=1.e-5):
    """
    """
    method = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = method.minimize(model['loss'])
    sess = tf.Session()
    init =tf.global_variables_initializer()
    sess.run(init)
    step =0 
    diff = np.inf
    pre_loss = np.inf
    while step<maxIter and diff>tol:
        _,loss,train_accuracy,y_pred = sess.run(
            [optimizer,model['loss'],model['accuracy'],model['y_pred']],
            feed_dict = {model['x']:X,model['y']:Y}
        )
        test_loss,test_accuracy = sess.run(
            [model['loss'],model['accuracy']],
            feed_dict = {model['x']:X_test,model['y']:Y_test}
        )
        diff = abs(pre_loss - loss)
        pre_loss = loss
        step += 1
        print('loss:{0}\tdiff:{1}\train_accuracy:{2},test_loss:{3}\ttest_accuracy:{4}'.format(loss,diff,train_accuracy,test_loss,test_accuracy))
