#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(125, 10, (9, 9))
plt.imshow(a, cmap=plt.get_cmap('gray'))
plt.colorbar(shrink=0.5)
plt.show()
