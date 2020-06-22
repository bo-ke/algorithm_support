# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataset_reader.py
@time: 2020/5/8 22:00

这一行开始写关于本文件的说明与解释


"""
import tensorflow as tf
import pandas as pd

WINDOW_SIZE = 8
df = pd.read_csv("./data/covid-19.csv", sep="\t")
df_data = df.set_index("date")
df_diff = df_data.diff(periods=1).dropna()
df_diff = df_diff.reset_index("date")

df_diff = df_diff.drop("date", axis=1).astype("float32")


def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    return dataset_batched


ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(df_diff.values, dtype=tf.float32)) \
    .window(WINDOW_SIZE, shift=1).flat_map(batch_dataset)

ds_label = tf.data.Dataset.from_tensor_slices(
    tf.constant(df_diff.values[WINDOW_SIZE:], dtype=tf.float32))

# 数据较小，可以将全部训练数据放入到一个batch中，提升性能
ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(38).cache()
