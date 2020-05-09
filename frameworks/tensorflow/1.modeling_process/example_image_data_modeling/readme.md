# 图片数据建模示例

- 使用数据: cifar2数据集  
    - 该数据集为cifar10数据集的子集，只包括前两种类别airplane和automobile。  
    - 该数据集具体信息：[https://wiki.kebo.site/books/dataset/page/cifar10](https://wiki.kebo.site/books/dataset/cifar10/cifar10)


- 定义模型方案：使用函数式API构建模型。

- 训练模型方案: 内置fit方法

- 保存模型方案: 使用TensorFlow原生方式保存模型