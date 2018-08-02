# # -*- coding: UTF-8 -*-

# 保证脚本与Python3兼容
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import matplotlib.pyplot as plt  
import numpy as np  
import tensorflow as tf  
from sklearn import datasets
import utils

def getData():
    """
    使用pandas读取数据
    """
    #read the data
    iris = datasets.load_iris()
    x_vals = np.array([[x[0], x[3]] for x in iris.data])
    y_vals = np.array([1.0 if y==0 else -1.0 for y in iris.target])
    return x_vals,y_vals



def create_linear_svm_model(dimenssion,alpha=0.01):
    """
    """
    np.random.seed(1024)
    x = tf.placeholder(tf.float64, shape=[None,dimenssion], name='x')
    y = tf.placeholder(tf.float64, shape=[None,1], name="y")
    A = tf.Variable(np.random.random([dimenssion,1]))
    b = tf.Variable(np.random.random([1,1]))
    z= tf.add(tf.matmul(x, A, name=None),b)#[None,1]
    m = tf.matmul(tf.transpose(z),y)
    loss = tf.reduce_mean(tf.maximum(np.float64(0),tf.subtract(np.float64(1.0),m)))
    model = {"loss": loss, "x": x,"y": y, "A": A,"b":b}
    return model

def gradientDescent(X,Y,model,learningRate=0.01,maxIter=10000,tol=1.e-5):
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
    print(X.shape,Y.shape)
    while step<maxIter and diff>tol:
        _,loss = sess.run(
            [optimizer,model['loss']],
            feed_dict = {model['x']:X,model['y']:Y}
        )
        diff = abs(pre_loss - loss)
        pre_loss = loss
        step += 1
        print('loss:{0}\tdiff:{1}'.format(loss,diff))

if __name__=='__main__':
    x_vals,y_vals = getData()
    x_train,y_train,x_test,y_test = utils.split_train_test(x_vals,y_vals)
    model = create_linear_svm_model(x_train.shape[1])
    gradientDescent(x_train,y_train.reshape(-1,1),model)
