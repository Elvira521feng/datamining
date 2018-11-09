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


def normalize_feature(df):  # 正则化
    """
    Applies function along input axis(default 0) of DataFrame.
    """
    # 特征缩放
    return df.apply(lambda column: (column - column.mean()) / column.std())


def linnear_regression(X_data, y_data, alpha, epoch, optimizer=tf.train.GradientDescentOptimizer):
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
            if len(loss_data) > 1 and np.abs(loss_data[-1]-loss_data[-2]) < 10 ** -9:
                # print('Converged at epoch {}'.format(i))
                break

    # clear the graph
    tf.reset_default_graph()
    return {'loss': loss_data, 'parameters': W_val}


def lr_cost(theta, X, y):
    m = X.shape[0]
    inner = X @ theta -y  # R(m*1)，X @ theta等价于X.dot(theta)
    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost


def gradient(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)

    return inner / m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
#   拟合线性回归，返回参数和代价
#     epoch: 批处理的轮数
#     """
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data
#批量梯度下降函数


# 返回一个类对象
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

# print(type(df))

# 看前五行
# print(data.head())
# 查看信息
# print(data.info())

# 散点图
# sns.lmplot('population', 'profit', data, size=6, fit_reg=False)
# plt.show()

X = get_X(data)
print(X.shape, type(X))
y = get_y(data)
print(y.shape, type(y))

theta = np.zeros(X.shape[1])  # X.shape[1]=2,代表特征数n

print(lr_cost(theta, X, y))
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)

print("final_theta", final_theta)
print('cost_data:', cost_data)
print(lr_cost(final_theta, X, y))

ax = sns.tsplot(cost_data, time=np.arange(epoch+1))
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()
#可以看到从第二轮代价数据变换很大，接下来平稳了

b = final_theta[0] # intercept，Y轴上的截距
m = final_theta[1] # slope，斜率

plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m + b, label="Prediction")
plt.legend(loc=2)
plt.show()