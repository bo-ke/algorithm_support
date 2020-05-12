# 文本数据建模示例

- 使用数据: imdb数据集  
    - 该数据集的目标是根据电影评论的文本内容预测评论的情感标签。  
    - 该数据集具体信息：[https://wiki.kebo.site/books/dataset/page/imdb](https://wiki.kebo.site/books/dataset/page/imdb)


- 定义模型方案：使用继承Model基类构建自定义模型

- 训练模型方案: 自定义训练循环训练模型

- 保存模型方案: 使用TensorFlow原生方式保存模型