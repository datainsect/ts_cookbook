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

def read_data(train_file_path,test_file_path):
    #set the path of the raw data
    data_path = os.path.join('/home/work/opdir/liuguilin1/.tensorflow/kaggle-avazu','base','data')
    train_file_path=os.path.join(data_path,train_file_path)
    test_file_path=os.path.join(data_path,test_file_path)
    train_df=pd.read_csv(train_file_path,sep=' ',header=None)
    test_df=pd.read_csv(test_file_path,sep=' ',header=None)
    # df= pd.concat((train_df,test_df),axis=0)
    return (train_df,test_df)


def split_X_Y(train_df,test_df):
	train_X = train_df[[x for x in xrange(2,17)]]
	train_Y = train_df[[1]]
	test_X = test_df[[x for x in xrange(2,17)]]
	test_Y = test_df[[1]]
	return (train_X,train_Y,test_X,test_Y)


def normalize_columns(m):
    max = m.max(axis=0)
    min = m.min(axis=0)
    return (m-min)/(max-min)

# def createLogisticModel(dimension):
#     """
#     """
#     np.random.seed(1024)
#     x = tf.placeholder(tf.float64, shape=[None, dimension], name='x')
#     y = tf.placeholder(tf.float64, shape=[None, 1], name="y")
#     A = tf.Variable(np.random.random([dimension, 1]))
#     b = tf.Variable(np.random.random([1,1]))
#     y_pred = tf.add(tf.matmul(x, A, name=None),b,name="y_pred")
#     loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y))
#     y_output = tf.round(tf.sigmoid(y_pred))
#     predictions_correct = tf.cast(tf.equal(y_output, y),tf.float32)
#     accuracy = tf.reduce_mean(predictions_correct)
#     model = {"loss": loss, "x": x,"y": y, "y_pred": y_pred, "A": A,"b":b,"accuracy":accuracy}
#     return model

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
    # error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y))
    error = tf.reduce_mean(tf.square(y-y_pred))
    loss = tf.add(error,l2_norm)
    y_output = tf.round(y_pred)
    predictions_correct = tf.cast(tf.equal(y_output, y),tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    model = {"loss": loss, "x": x,"y": y, "y_pred": y_pred, "A": A,"b":b,"accuracy":accuracy}
    return model

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
        test_accuracy = sess.run(
            model['accuracy'],
            feed_dict = {model['x']:X_test,model['y']:Y_test}
        )
        diff = abs(pre_loss - loss)
        pre_loss = loss
        step += 1
        print('loss:{0}\tdiff:{1}\train_accuracy:{2},test_accuracy:{3}'.format(loss,diff,train_accuracy,test_accuracy))
        #print(y_pred.transpose())
    

if __name__ == "__main__":
    train_df,test_df =  read_data('tr.rx.app.sp','va.rx.app.sp')
    x_train,y_train,x_test,y_test = split_X_Y(train_df,test_df)
    k=5
    x_train=normalize_columns(x_train) 
    x_train.drop([12],axis=1) 
    model = createFMModel(x_train.shape[1],k)
    gradientDescent(x_train,y_train,x_test,y_test,model,learningRate=0.000001)
