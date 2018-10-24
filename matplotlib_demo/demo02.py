#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

import numpy as np
from matplotlib.cbook import get_sample_data
import matplotlib.pyplot as plt

data = 2 * np.random.random((10, 10))
data2 = 3 * np.random.random((10, 10))
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
plt.plot((X, Y),U)
plt.show()
