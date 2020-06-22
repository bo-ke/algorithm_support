# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: model.py
@time: 2020/4/26 0:05

这一行开始写关于本文件的说明与解释


"""
import json
import tensorflow as tf

from dataset_reader import MAX_LEN, MAX_WORDS, ds_train, ds_test


class CnnModel(tf.keras.models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv_1 = tf.keras.layers.Conv1D(16, kernel_size=5, name="conv_1", activation="relu")
        self.pool = tf.keras.layers.MaxPool1D()
        self.conv_2 = tf.keras.layers.Conv1D(128, kernel_size=2, name="conv_2", activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class Trainer:
    def __init__(self):
        self.model = CnnModel()
        self.model.build(input_shape=(None, MAX_LEN))
        self.model.summary()

        self.optimizer = tf.keras.optimizers.Nadam()
        self.loss_func = tf.keras.losses.BinaryCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.valid_metric = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')

    # 打印时间分割线
    @classmethod
    @tf.function
    def print_bar(cls):
        today_ts = tf.timestamp() % (24 * 60 * 60)

        hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
        mini_te = tf.cast((today_ts % 3600) // 60, tf.int32)
        second = tf.cast(tf.floor(today_ts % 60), tf.int32)

        def time_format(m):
            if tf.strings.length(tf.strings.format("{}", m)) == 1:
                return tf.strings.format("0{}", m)
            else:
                return tf.strings.format("{}", m)

        time_string = tf.strings.join([time_format(hour), time_format(mini_te),
                                       time_format(second)], separator=":")
        tf.print("==========" * 8 + time_string)

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            loss = self.loss_func(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_metric.update_state(labels, predictions)

    @tf.function
    def valid_step(self, features, labels):
        predictions = self.model(features, training=False)
        batch_loss = self.loss_func(labels, predictions)
        self.valid_loss.update_state(batch_loss)
        self.valid_metric.update_state(labels, predictions)

    def train_model(self, epochs, train_data, valid_data):
        for epoch in tf.range(1, epochs + 1):

            for features, labels in train_data:
                self.train_step(features, labels)

            for features, labels in valid_data:
                self.valid_step(features, labels)

            # 此处logs模板需要根据metric具体情况修改
            logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

            if epoch % 1 == 0:
                self.print_bar()
                tf.print(tf.strings.format(logs,
                                           (epoch, self.train_loss.result(), self.train_metric.result(),
                                            self.valid_loss.result(),
                                            self.valid_metric.result())))
                tf.print("")

            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            self.train_metric.reset_states()
            self.valid_metric.reset_states()
        # self.model.save_weights('./data/output/keras_model_weight.h5')
        # model_json = self.model.to_json()
        # json.dump(model_json, open("./data/output/model_json.json", "w"), indent=4)
        self.model.save('./data/output', save_format="tf")

        print('export saved model.')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_model(epochs=5, train_data=ds_train, valid_data=ds_test)
