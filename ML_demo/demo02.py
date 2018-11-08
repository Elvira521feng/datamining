#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

# import numpy as np
# import matplotlib.pyplot as plt
#
# a = [[1, 1.8], [2, 4.1], [1.5, 3.2], [3, 5.8]]
#
# t0 = 0
# t1 = 0
#
# t0
# x = np.arange(0, 8, 0.2)
#
# y = 2 * x
# plt.plot(x,y)
# plt.show()

# -*- coding: utf-8 -*-
"""
__author__ = 'ljyn4180'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 代价函数
def CostFunction(matrixX, matrixY, matrixTheta):
    Inner = np.power(((matrixX * matrixTheta.T) - matrixY), 2)
    return np.sum(Inner) / (2 * len(matrixX))


# 梯度下降迭代函数
def GradientDescent(matrixX, matrixY, matrixTheta, fAlpha, nIterCounts):
    matrixThetaTemp = np.matrix(np.zeros(matrixTheta.shape))
    nParameters = int(matrixTheta.ravel().shape[1])
    arrayCost = np.zeros(nIterCounts)

    for i in xrange(nIterCounts):
        matrixError = (matrixX * matrixTheta.T) - matrixY

        for j in xrange(nParameters):
            matrixSumTerm = np.multiply(matrixError, matrixX[:, j])
            matrixThetaTemp[0, j] = matrixTheta[0, j] - fAlpha / len(matrixX) * np.sum(matrixSumTerm)

        matrixTheta = matrixThetaTemp
        arrayCost[i] = CostFunction(matrixX, matrixY, matrixTheta)

    return matrixTheta, arrayCost


# 显示线性回归结果
def ShowLineRegressionResult(dataFrame, matrixTheta):
    x = np.linspace(dataFrame.Population.min(), dataFrame.Population.max(), 100)
    f = matrixTheta[0, 0] + (matrixTheta[0, 1] * x)

    plt.subplot(221)
    plt.plot(x, f, 'r', label='Prediction')
    plt.scatter(dataFrame.Population, dataFrame.Profit, label='Training Data')
    plt.legend(loc=2)
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.title('Predicted Profit vs. Population Size')


# 显示代价函数的值的变化情况
def ShowCostChange(arrayCost, nIterCounts):
    plt.subplot(222)
    plt.plot(np.arange(nIterCounts), arrayCost, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs. Training Epoch')


# 显示不同Alpha值得学习曲线
def ShowLearningRateChange(dictCost, nIterCounts):
    plt.subplot(223)
    for fAlpha, arrayCost in dictCost.iteritems():
        plt.plot(np.arange(nIterCounts), arrayCost, label=fAlpha)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('learning rate')


# 正规方程
def NormalEquation(matrixX, matrixY):
    matrixTheta = np.linalg.inv(matrixX.T * matrixX) * matrixX.T * matrixY  # Python2.7
    # matrixTheta = np.linalg.inv(matrixX.T @ matrixX) @ matrixX.T @ matrixY  # Python3
    return matrixTheta


# 标准化特征值
def NormalizeFeature(dataFrame):
    return dataFrame.apply(lambda column: (column - column.mean()) / column.std())


def ExerciseOne():
    path = 'ex1data1.txt'
    dataFrame = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    # 补项
    dataFrame.insert(0, 'Ones', 1)
    nColumnCount = dataFrame.shape[1]
    dataFrameX = dataFrame.iloc[:, 0:nColumnCount - 1]
    dataFrameY = dataFrame.iloc[:, nColumnCount - 1:nColumnCount]

    # 初始化数据
    matrixX = np.matrix(dataFrameX.values)
    matrixY = np.matrix(dataFrameY.values)
    matrixOriginTheta = np.matrix(np.zeros(dataFrameX.shape[1]))

    # 设置学习率和迭代次数
    fAlpha = 0.01
    nIterCounts = 1000
    matrixTheta, arrayCost = GradientDescent(matrixX, matrixY, matrixOriginTheta, fAlpha, nIterCounts)

    print(matrixTheta)
    print(NormalEquation(matrixX, matrixY))

    # 设置不同的学习率
    arrayAlpha = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    dictCost = {}
    for fAlpha in arrayAlpha:
        _, arrayCostTemp = GradientDescent(matrixX, matrixY, matrixOriginTheta, fAlpha, nIterCounts)
        dictCost[fAlpha] = arrayCostTemp

    # 显示不同图表
    plt.figure(figsize=(12, 12))
    ShowLineRegressionResult(dataFrame, matrixTheta)
    ShowCostChange(arrayCost, nIterCounts)
    ShowLearningRateChange(dictCost, nIterCounts)
    plt.show()


ExerciseOne()

