#!/usr/bin/env python
#-*- coding:utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np


def computer_cost(x, y, theta):
    inner = np.power(((x*theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


x1 = []
y1 = []
theta1 = []
J = computer_cost(x1, y1, theta1)

fig = plt.figure()
ax = Axes3D(fig)

plt.plot(x1, y1, theta1, J)
plt.show()





