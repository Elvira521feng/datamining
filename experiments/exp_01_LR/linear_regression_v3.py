#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def linear_regression(X_data, y_data, alpha, epoch, optimizer=tf.train.GradientDescentOptimizer):
    X = tf.placeholder(tf.float32, shape=X_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)

    with tf.variable_scope('linear-regression'):
        W = tf.get_variable('weights', (X_data.shape[1], 1), initializer=tf.constant_initializer())
        # 矩阵相乘
        y_pred = tf.matmul(X, W)
        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)

    opt = optimizer(learning_rate=alpha)
    opt_operation = opt.minimize(loss)

    # run the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_data = []

        for i in range(epoch):
            _,loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: X_data, y:y_data})
            loss_data.append(loss_val[0,0])
            if len(loss_data) > 1 and np.abs(loss_data[-1] - loss_data[-2]) < 10 ** -9:
                # print('Converged at epoch {}'.format(i))
                break

    # clear the graph
    tf.reset_default_graph()
    return {'loss': loss_data, 'parameters': W_val}


raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
print(raw_data.head())

data = normalize_feature(raw_data)
# data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
print(data.head())

X = get_X(data)
print(X.shape, type(X))

y = get_y(data).reshape(len(X), 1)  # special treatment for tensorflow input data
print(y.shape, type(y))

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)#感觉和批量梯度下降的theta的值有点差距
print(final_theta2)

epoch = 2000
alpha = 0.01

optimizer_dict={'GD': tf.train.GradientDescentOptimizer,
                'Adagrad': tf.train.AdagradOptimizer,
                'Adam': tf.train.AdamOptimizer,
                'Ftrl': tf.train.FtrlOptimizer,
                'RMS': tf.train.RMSPropOptimizer
               }

results = []

for name in optimizer_dict:
    # print(name)
    res = linear_regression(X, y, alpha, epoch, optimizer=optimizer_dict[name])
    res['name'] = name
    results.append(res)

fig, ax = plt.subplots(figsize=(16, 9))

for res in results:
    # print(res['loss'])
    loss_data = res['loss']

    #     print('for optimizer {}'.format(res['name']))
    #     print('final parameters\n', res['parameters'])
    #     print('final loss={}\n'.format(loss_data[-1]))
    ax.plot(np.arange(len(loss_data)), loss_data, label=res['name'])

# res = results[0]
# loss_data = res['loss']
# ax.plot(np.arange(len(loss_data)), loss_data, label=res['name'])

ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('different optimizer', fontsize=18)
plt.show()
