#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    plt.plot(x, y)

    ax = plt.gca()  # get current axis 获得坐标轴对象

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # ax.xaxis.set_ticks_position("left")
    # ax.spines['left'].set_position(('data', 0))

    plt.show()


if __name__ == '__main__':
   plot_sigmoid()

