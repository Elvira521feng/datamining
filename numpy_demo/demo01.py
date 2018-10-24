#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Elvira

import numpy as np

# 创建简单的列表
a = [1, 2, 3, 4]
# 将列表转换成数组
b = np.array(a)
# 直接创建数组
c = np.array([2, 3, 4, 5])
print(type(a), type(b), type(c))
d = np.arange(10, 25, 5)
print(d)
# 数组元素个数
print(b.size)
# 数组形状
print(b.shape)
# 数组维度
print(b.ndim)
# 数组元素类型
print(b.dtype)

# N维数组
# 10行10列数值为浮点1的矩阵
array_one = np.ones([10,10])
print(array_one)

# 深拷贝--array
# 浅拷贝--asarray

# 随机数组
# 均匀分布
np.random.rand(10, 10) # 10行10列,默认范围在0-1之间
# 指定范围内的随机数
uniform = np.random.uniform(0, 100)
print(uniform)
# 指定范围内的随机整数
np.random.randint(0, 100)
# 指定范围内的矩阵
randint = np.random.randint(0, 100, [5, 5])
print(randint)
print(type(randint))
# 给定均值/标准差/维度的正态分布矩阵
normal = np.random.normal(1.75, 0.1, (5, 6))
print(normal)

# 数组索引,切片
after_normal = normal[1:3, 1:4] # 2-3行,2-4列
print(after_normal)

# 一维转多维
one_20 = np.ones([20])
print(one_20)
one_5_4 = one_20.reshape([5,4])
print(one_5_4)

# 拉平数组
print(after_normal.ravel())

# 条件运算
stus_score = np.random.normal(80, 10, [5, 2])
print(stus_score)
print(stus_score > 80)  # 判断结果

# 指定轴计算(amax:最大值,amin:最小值,mean:平均值,std:方差,cumsum:
print(np.amax(stus_score, axis=0))  # 每一列的最大值
print(np.amin(stus_score, axis=1))  # 每一行的最大值

# 数组与数的运算 + - * /
stus_score[:,0] = stus_score[:,0] + 5  # 第一列每个数+5
print(stus_score)

# 数组间运算
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a + b
d = a - b
e = a * b
f = a / b
print("a+b为", c)
print("a-b为", d)
print("a*b为", e)
print("a/b为", f)

# 矩阵乘法 dot
q = np.array([[0.4], [0.6]])
result = np.dot(stus_score, q)
print("最终结果为:")
print(result)

# 矩阵拼接
print("v1为:")
v1 = [[0, 1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10, 11]]
print(v1)
print("v2为:")
v2 = [[12, 13, 14, 15, 16, 17],
      [18, 19, 20, 21, 22, 23]]
print(v2)

# 拼接
print(np.concatenate((v1, v2), axis=0))

# 垂直拼接
result = np.vstack((v1, v2))
print("v1和v2垂直拼接的结果为")
print(result)
print(np.r_[v1, v2])

# 水平拼接
result = np.hstack((v1, v2))
print("v1和v2水平拼接的结果为")
print(result)
print(np.c_[v1, v2])
