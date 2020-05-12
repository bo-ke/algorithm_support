# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: itachi971009@gmail.com

@version: 1.0
@file: model.py
@time: 2020/5/8 22:05

这一行开始写关于本文件的说明与解释


"""
import tensorflow as tf
import os
import datetime

from dataset_reader import ds_train

output_dir = os.path.join(os.path.abspath("./data"), "output")


# 考虑到新增确诊，新增治愈，新增死亡人数数据不可能小于0，设计如下结构
class Block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)

    def call(self, x_input, x):
        x_out = tf.maximum((1 + x) * x_input[:, -1, :], 0.0)
        return x_out

    def get_config(self):
        config = super(Block, self).get_config()
        return config


tf.keras.backend.clear_session()
x_input = tf.keras.layers.Input(shape=(None, 3), dtype=tf.float32)
x = tf.keras.layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x_input)
x = tf.keras.layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
x = tf.keras.layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
x = tf.keras.layers.LSTM(3, input_shape=(None, 3))(x)
x = tf.keras.layers.Dense(3)(x)

# 考虑到新增确诊，新增治愈，新增死亡人数数据不可能小于0，设计如下结构
# x = tf.maximum((1+x)*x_input[:,-1,:],0.0)
x = Block()(x_input, x)
model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
model.summary()


# 自定义损失函数，考虑平方差和预测目标的比值
class MSPE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        err_percent = (y_true - y_pred) ** 2 / (tf.maximum(y_true ** 2, 1e-7))
        mean_err_percent = tf.reduce_mean(err_percent)
        return mean_err_percent

    def get_config(self):
        config = super(MSPE, self).get_config()
        return config


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=MSPE(name="MSPE"))

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(output_dir, stamp)

# 在 Python3 下建议使用 pathlib 修正各操作系统的路径
# from pathlib import Path
# stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = str(Path('./data/autograph/' + stamp))

tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# 如果loss在100个epoch后没有提升，学习率减半。
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=100)
# 当loss在200个epoch后没有提升，则提前终止训练。
stop_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=200)
callbacks_list = [tb_callback, lr_callback, stop_callback]

history = model.fit(ds_train, epochs=500, callbacks=callbacks_list)

model.save(output_dir, save_format="tf")
print('export saved model.')
