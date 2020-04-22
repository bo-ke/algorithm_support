# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: itachi971009@gmail.com

@version: 1.0
@file: predictor.py
@time: 2020/4/23 0:25

这一行开始写关于本文件的说明与解释


"""
import tensorflow as tf

model_path = "./data/keras_model.h5"


class Predictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, x):
        return self.model.predict(x)
