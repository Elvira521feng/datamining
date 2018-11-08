1.DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)  在给定轴方向应用函数

参数：
func : function|要应用在行和列的函数
axis : {0 or ‘index’, 1 or ‘columns’}, default 0|选择是行还是列
broadcast : boolean, default False|For aggregation functions, return object of same size with values propagated
raw : boolean, default False|If False, convert each row or column into a Series. If raw=True the passed function will receive ndarray objects instead.
reduce : boolean or None, default None|Try to apply reduction procedures.
args : tuple|函数的参数
------------------------------------------------------------------------------------
2.矩阵乘法
tf.matmul()
np.dot()