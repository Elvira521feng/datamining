#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

from matplotlib import pyplot as plt

X = [[1, 2], [3, 4], [5, 6]]
plt.imshow(X)
plt.colorbar(shrink=0.5)
plt.show()