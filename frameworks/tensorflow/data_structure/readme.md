# TensorFlow中的数据结构

TensorFlow中的基本数据结构是张量Tensor。张量即多维数组。
TensorFlow中的张量煜numpy中的array很类似

张量主要有两种，常量constant于变量Variable
> 常量在计算图中不可以被重新赋值，变量可以在计算题中用assign等算子赋值

## 张量的数据类型
张量的数据类型和numpy.array基本一一对应
```python
import numpy as np
import tensorflow as tf

i = tf.constant(1) # tf.int32 类型常量
l = tf.constant(1,dtype = tf.int64) # tf.int64 类型常量
f = tf.constant(1.23) #tf.float32 类型常量
d = tf.constant(3.14,dtype = tf.double) # tf.double 类型常量
s = tf.constant("hello world") # tf.string类型常量
b = tf.constant(True) #tf.bool类型常量


print(tf.int64 == np.int64) 
print(tf.bool == np.bool)
print(tf.double == np.float64)
print(tf.string == np.unicode) # tf.string类型和np.unicode类型不等价
```

```shell script
True
True
True
False
```

## 不同类型的数据表示方法
不同类型的数据可以用不同维度的张量来表示  
- 标量: 0维张量
- 向量：1维张量  
- 矩阵：2维张量
- 彩色图像：有rgb3通道，可表示为3维张量
- 视频：相对图像还有时间维，可表示为4维张量

## 获取张量的维度
可以简单的总结为：有几层中括号，就是多杀维的张量  
可以通过`tf.rank()`跟`tensor.numpy().ndim`得到

## 变量张量
模型中需要被训练的参数一般被设置为变量
```python
# 常量值不可以改变，常量的重新赋值相当于创造新的内存空间
import tensorflow as tf
c = tf.constant([1.0, 2.0])
print(c)
print(id(c))
c = c+tf.constant([1.0, 1.0])
print(c)
print(id(c))
```

```shell script
tf.Tensor([1. 2.], shape=(2,), dtype=float32)
5276289568
tf.Tensor([2. 3.], shape=(2,), dtype=float32)
5276290240
```

```python
# 变量的值可以改变，可以通过assign, assign_add等方法给变量重新赋值
import tensorflow as tf
v = tf.Variable([1.0, 2.0], name="v")
print(v)
print(id(v))
v.assign([1.0, 2.0])
print(v)
print(id(v))
```

```shell script
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([1., 2.], dtype=float32)>
5276259888
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([2., 3.], dtype=float32)>
5276259888
```