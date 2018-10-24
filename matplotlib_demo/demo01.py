#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)

fig = plt.figure()
fig.add_axes()
ax1 = fig.add_subplot(221)

plt.plot(x, y)
plt.sca(ax1)

ax2 = fig.add_subplot(220)


plt.show()
