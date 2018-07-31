# # -*- coding: UTF-8 -*-
"""
此脚本用于展示逻辑回归模型的搭建过程
"""


# 保证脚本与Python3兼容
from __future__ import print_function

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
    data['beta'] = 1
    data["label_code"] = pd.Categorical(data.label).codes
    x_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", 'beta']
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
    return(x_vals_train,y_vals_train,x_vals_test,y_vals_test)

def normalize_columns(m):
    max = m.max(axis=0)
    min = m.min(axis=0)
    (m-min)/(max-min)

def createLogisticModel(num,dimension):
    """
    """
    np.random.seed(1024)
    # 定义自变量和应变量
    x = tf.placeholder(tf.float64, shape=[num, dimension], name='x')
    ## 将被预测值写成矩阵形式，会极大加快速度
    y = tf.placeholder(tf.float64, shape=[num, 1], name="y")
    # 定义参数估计值和预测值
    betaPred = tf.Variable(np.random.random([dimension, 1]))
    y_pred = tf.matmul(x, betaPred, name="y_pred")
    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y))
    model = {"loss": loss, "x": x,"y": y, "y_pred": y_pred, "paras": betaPred}
    return model

def gradientDescent(X,Y,model,learningRate=0.01,maxIter=10000,tol=1.e-1):
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
        _,loss = sess.run(
            [optimizer,model['loss']],
            feed_dict = {model['x']:X,model['y']:Y}
        )
        diff = pre_loss - loss
        pre_loss = loss
        step += 1
        if step%25==0 :
            print('loss:{0}\tdiff:{1}'.format(loss,diff))
    print(model['loss'].eval(session=sess))


if __name__ == "__main__":
    dataPath = "data/adult.data"
    x_vals,y_vals=readData(dataPath)
    x_train,y_train,x_test,y_test = split_train_test(x_vals,y_vals)
    x_train = normalize_columns(x_train)
    x_test = normalize_columns(x_test)
    num = x_train.shape[0]
    dimension = x_train.shape[1]
    model = createLogisticModel(num,dimension)
    gradientDescent(x_train,y_train,model)