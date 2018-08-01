# # -*- coding: UTF-8 -*-


# 保证脚本与Python3兼容
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
import pandas as pd
import linear_reg as lr


def readData(path):
    """
    使用pandas读取数据
    """
    #read the data
    data = pd.read_csv(path)
    data["label_code"] = pd.Categorical(data.label).codes
    x_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    y_clos = ["label_code"]
    x_vals = data[x_cols]
    y_vals = data[y_clos]
    return (x_vals,y_vals)

def split_train_test(x_vals,y_vals):
    """
    split the data into test and train
    """
    train_indices = np.random.choice(len(x_vals),int(round(len(x_vals)*0.8)), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals.ix[train_indices]
    x_vals_test = x_vals.ix[test_indices]
    y_vals_train = y_vals.ix[train_indices]
    y_vals_test = y_vals.ix[test_indices]
    return (x_vals_train,y_vals_train,x_vals_test,y_vals_test)

def normalize_columns(m):
    max = m.max(axis=0)
    min = m.min(axis=0)
    return (m-min)/(max-min)

def createLogisticModel(dimension):
    """
    """
    np.random.seed(1024)
    x = tf.placeholder(tf.float64, shape=[None, dimension], name='x')
    y = tf.placeholder(tf.float64, shape=[None, 1], name="y")
    A = tf.Variable(np.random.random([dimension, 1]))
    b = tf.Variable(np.random.random([1,1]))
    y_pred = tf.add(tf.matmul(x, A, name=None),b,name="y_pred")
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y))
    y_output = tf.round(tf.sigmoid(y_pred))
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
        _,loss,train_accuracy = sess.run(
            [optimizer,model['loss'],model['accuracy']],
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
    

if __name__ == "__main__":
    dataPath = "data/adult.data"
    x_vals,y_vals=readData(dataPath)
    x_train,y_train,x_test,y_test = split_train_test(x_vals,y_vals)
    x_train = normalize_columns(x_train)
    x_test = normalize_columns(x_test)
    model = createLogisticModel(x_train.shape[1])
    gradientDescent(x_train,y_train,x_test,y_test,model)