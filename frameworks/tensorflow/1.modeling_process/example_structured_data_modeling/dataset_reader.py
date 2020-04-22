# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: itachi971009@gmail.com

@version: 1.0
@file: dataset_reader.py
@time: 2020/4/23 0:21

这一行开始写关于本文件的说明与解释


"""
import pandas as pd

df_train_raw = pd.read_csv('./data/titanic/train.csv')
df_test_raw = pd.read_csv('./data/titanic/test.csv')


class DatasetReader:
    @classmethod
    def pre_processing(cls, df_data):
        df_result = pd.DataFrame()

        # Pclass
        df_Pclass = pd.get_dummies(df_data['Pclass'])
        df_Pclass.columns = ['Pclass_' + str(x) for x in df_Pclass.columns]
        df_result = pd.concat([df_result, df_Pclass], axis=1)

        # Sex
        df_Sex = pd.get_dummies(df_data['Sex'])
        df_result = pd.concat([df_result, df_Sex], axis=1)

        # Age
        df_result['Age'] = df_data['Age'].fillna(0)
        df_result['Age_null'] = pd.isna(df_data['Age']).astype('int32')

        # SibSp,Parch,Fare
        df_result['SibSp'] = df_data['SibSp']
        df_result['Parch'] = df_data['Parch']
        df_result['Fare'] = df_data['Fare']

        # Cabin
        df_result['Cabin_null'] = pd.isna(df_data['Cabin']).astype('int32')

        # Embarked
        df_embarked = pd.get_dummies(df_data['Embarked'], dummy_na=True)
        df_embarked.columns = ['Embarked_' + str(x) for x in df_embarked.columns]
        df_result = pd.concat([df_result, df_embarked], axis=1)

        return df_result


x_train = DatasetReader.pre_processing(df_data=df_train_raw)
y_train = df_train_raw['Survived'].values

x_test = DatasetReader.pre_processing(df_data=df_test_raw)
y_test = df_test_raw['Survived'].values

print("x_train.shape =", x_train.shape)
print("x_test.shape =", x_test.shape)
