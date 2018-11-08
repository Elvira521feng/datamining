#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

# x, y, z 均为 0 到 1 之间的 100 个随机数
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x, y, z)

plt.show()

