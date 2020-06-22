# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataset_reader.py
@time: 2020/4/26 0:05

这一行开始写关于本文件的说明与解释


"""
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string

train_data_path = "./data/imdb/train.csv"
test_data_path = "./data/imdb/test.csv"

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20


# 构建管道
def split_line(line):
    arr = tf.strings.split(line, "\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)
    return text, label


ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames=[test_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)


# 构建词典
def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,
                                                   '[%s]' % re.escape(string.punctuation), '')
    return cleaned_punctuation


vectorize_layer = TextVectorization(
    standardize=clean_text,
    split='whitespace',
    max_tokens=MAX_WORDS - 1,  # 有一个留给占位符
    output_mode='int',
    output_sequence_length=MAX_LEN)

ds_text = ds_train_raw.map(lambda text, label: text)
vectorize_layer.adapt(ds_text)
print(vectorize_layer.get_vocabulary()[0:100])

# 单词编码
ds_train = ds_train_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
