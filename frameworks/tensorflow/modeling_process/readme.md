# 建模流程

## 准备数据
- 结构化数据:   
对于结构化数据，一般会使用Pandas中的DataFrame进行预处理。利用Pandas的数据可视化功能我们可以简单地进行探索性数据分析EDA（Exploratory Data Analysis）
- 图片数据:  
对于图片数据，在tensorflow中有两种方案  
    - 使用tf.keras中的ImageDataGenerator工具构建图片数据生成器。可参考文章：
    [https://zhuanlan.zhihu.com/p/67466552](https://zhuanlan.zhihu.com/p/67466552)
    - 使用TensorFlow的原生方法，更加灵活，使用得当的话也可以获得更好的性能。在图像数据建模example中，我们使用此方法。
- 文本数据:  
对于文本数据预处理，常用方案有两种
    - 利用tf.keras.preprocessing中的Tokenizer词典构建工具和tf.keras.utils.Sequence构建文本数据生成器管道。
    - 使用tf.data.Dataset搭配.keras.layers.experimental.preprocessing.TextVectorization预处理层。

## 定义模型
使用Keras接口有以下3种方式构建模型：
- 使用Sequential按层顺序构建模型
- 使用函数式API构建任意结构模型
- 继承Model基类构建自定义模型。

## 训练模型
训练模型通常有3种方法
- 内置fit方法
- 内置train_on_batch方法
- 自定义训练循环

## 保存模型
保存模型有两种方法
- 使用Keras方式保存模型（该方案仅适合使用Python环境恢复模型）
    - 保存模型结构及权重到同一个文件
    - 分别保存模型结构与权重
- 使用TensorFlow原生方式保存（可以跨平台部署）

## examples
[1. 结构化数据建模示例](https://github.com/whu-SpongeBob/algorithm_support/tree/master/frameworks/tensorflow/1.modeling_process/example_structured_data_modeling)  
[2. 文本数据建模示例](https://github.com/whu-SpongeBob/algorithm_support/tree/master/frameworks/tensorflow/1.modeling_process/example_text_data_modeling)  
[3. 序列化数据建模示例](https://github.com/whu-SpongeBob/algorithm_support/tree/master/frameworks/tensorflow/1.modeling_process/example_temporal_sequences_modeling)