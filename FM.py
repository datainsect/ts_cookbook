# # -*- coding: UTF-8 -*-
#https://blog.csdn.net/jiangjiang_jian/article/details/80630873

# 保证脚本与Python3兼容
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


import numpy as np
import tensorflow as tf
import pandas as pd
import os
from utils import gradientDescent,read_data


def split_X_Y(train_df,test_df):
    n = train_df.columns.size
    train_X = train_df[[x for x in xrange(0,n-1)]]
    train_Y = train_df[[n-1]]
    test_X = test_df[[x for x in xrange(0,n-1)]]
    test_Y = test_df[[n-1]]
    return (train_X,train_Y,test_X,test_Y)

def split_X_Y_sp(train_df,test_df):
    n = train_df.columns.size
    train_X = train_df[[x for x in xrange(2,n)]]
    train_Y = train_df[[1]]
    test_X = test_df[[x for x in xrange(2,n)]]
    test_Y = test_df[[1]]
    return (train_X,train_Y,test_X,test_Y)

def normalize_columns(m):
    max = m.max(axis=0)
    min = m.min(axis=0)
    return (m-min)/(max-min)

def createFMModel(dimension,latent_factor):
    """
    """
    np.random.seed(1024)
    x = tf.placeholder(tf.float64, shape=[None, dimension], name='x')
    y = tf.placeholder(tf.float64, shape=[None, 1], name="y")
    A = tf.Variable(np.random.random([dimension, 1]))
    b = tf.Variable(np.random.random([1,1]))
    V = tf.Variable(np.random.random([dimension,latent_factor])
    )
    linear_terms = tf.add(tf.matmul(x, A, name=None),b)
    interactions = (tf.multiply(tf.cast(0.5,tf.float64),
                tf.reduce_sum(
                    tf.subtract(
                        tf.pow( tf.matmul(x, V), 2),
                        tf.matmul(tf.pow(x, 2), tf.pow(V, 2))),
                    1, keepdims=True)))
    
    y_pred = tf.add(linear_terms,interactions)
    
    # lambda_w = tf.constant(tf.cast(0.001,tf.float64),name='lambda_w')
    # lambda_v = tf.constant(tf.cast(0.001,tf.float64),name='lambda_v')
    l2_norm = tf.reduce_sum(
      tf.add(
        tf.multiply(tf.cast(0.001,tf.float64),tf.pow(A,2)),
        tf.multiply(tf.cast(0.001,tf.float64),tf.pow(V,2))
      )
    )
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y))
    # error = tf.reduce_mean(tf.square(y-y_pred))
    loss = tf.add(error,l2_norm)
    y_output = tf.round(y_pred)
    predictions_correct = tf.cast(tf.equal(y_output, y),tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    model = {"loss": error, "x": x,"y": y, "y_pred": y_pred, "A": A,"b":b,"accuracy":accuracy}
    return model

def FM_on_diabetes():
    train_df,test_df =  read_data('diabetes_train.txt','diabetes_test.txt')
    x_train,y_train,x_test,y_test = split_X_Y(train_df,test_df)
    k=5
    x_train=normalize_columns(x_train) 
    x_test=normalize_columns(x_test) 
    model = createFMModel(x_train.shape[1],k)
    gradientDescent(x_train,y_train,x_test,y_test,model,learningRate=0.0004,maxIter=100000,tol=1.e-7)   


def FM_on_sp(learningRate=0.00000001,k=5,maxIter=100000,tol=1.e-7):
    train_df,test_df =  read_data('tr.rx.app.sp','va.rx.app.sp',sep=' ')
    x_train,y_train,x_test,y_test = split_X_Y_sp(train_df,test_df)
    x_train=normalize_columns(x_train)
    x_train = x_train.drop([12],axis=1) 
    x_test=normalize_columns(x_test)
    x_test = x_test.drop([12],axis=1) 
    model = createFMModel(x_train.shape[1],k)
    gradientDescent(x_train,y_train,x_test,y_test,model,learningRate,maxIter,tol)   


if __name__ == "__main__":
    FM_on_sp(learningRate=0.0001,k=5)
