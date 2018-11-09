#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sns.set('notebook', 'whitegrid', 'dark')


def get_X(df):  # 读取特征
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].as_matrix()  # 这个操作返回 ndarray,不是矩阵


def get_y(df):  # 读取标签

    """
    assume the last column is the target
    """
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())


def lr_cost(theta, X, y):
    # """
    #     X: R(m*n), m 样本数, n 特征数
    #     y: R(m)
    #     theta : R(n), 线性回归的参数
    # """
    m = X.shape[0]
    inner = X @ theta -y
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost


def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m


def batch_gradient_decent(theta, X, y,epoch, alpha):
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()
    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta,cost_data

raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
print(raw_data.head())

data = normalize_feature(raw_data)
print(data.head())

X = get_X(data)
print(X.shape, type(X))

y = get_y(data)
print(y.shape, type(y))  # 看下数据的维度和类型

alpha = 0.01  # 学习率
theta = np.zeros(X.shape[1])  # X.shape[1]：特征数n
epoch = 500  # 轮数

final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)

base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base*3)))
print(candidate)

epoch=50

fig, ax = plt.subplots(figsize=(16, 9))

for alpha in candidate:
    _, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
    ax.plot(np.arange(epoch+1), cost_data, label=alpha)

ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('learning rate', fontsize=18)
plt.show()