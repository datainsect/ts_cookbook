# # -*- coding: UTF-8 -*-
"""
此脚本用于展示逻辑回归模型的搭建过程
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd

def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    data['beta'] = 1
    cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", 'beta',"label"]
    return data[cols]

def transLabel(data):
    """
    """
    data["label_code"] = pd.Categorical(data.label).codes
    data.drop(columns=['label'])
    return data

def createLogisticModel(dimension):
    """
    搭建模型，包括数据中的自变量，应变量和损失函数

    参数
    ----
    dimension : int，自变量的个数

    返回
    ----
    model ：dict，里面包含模型的参数，损失函数，自变量，应变量
    """
    np.random.seed(1024)
    # 定义自变量和应变量
    x = tf.placeholder(tf.float64, shape=[None, dimension+1], name='x')
    ## 将被预测值写成矩阵形式，会极大加快速度
    y = tf.placeholder(tf.float64, shape=[None, 1], name="y")
    # 定义参数估计值和预测值
    betaPred = tf.Variable(np.random.random([dimension+1, 1]))
    yPred = tf.matmul(x, betaPred, name="y_pred")
    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ｙPred, y_target))
    model = {"loss_function": loss, "independent_variable": x,
        "dependent_variable": y, "prediction": yPred, "model_params": betaPred}
    return model

if __name__ == "__main__":
    if os.name == "nt":
        dataPath = "%s\\data\\adult.data" % homePath
    else:
        dataPath = "%s/data/adult.data" % homePath
    data = readData(dataPath)
    data = transLabel(data)