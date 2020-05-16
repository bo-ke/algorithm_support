# TensorFlow中的计算图

在tensorflow中,有三种计算图的构建方式  

- 静态计算图： 在tensorflow1.0的时代使用的是静态计算图，需要先使用tensorflow的各种算子创建计算图，然后再开启一个会话session，显示执行计算图
- 动态计算图： 在tensorflow2.0时代，采用的是动态计算图，即每使用一个算子后，该算子会被动态加入到隐含的默认计算图中立即执行得到结果，而无需开启session
- Autograph