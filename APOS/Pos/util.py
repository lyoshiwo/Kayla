#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
import xgboost as xgb


def covert_csv_to_svm(csv_name, svm_name):
    oct27 = pd.read_csv(csv_name)
    list_oct27_feature = []
    list_label = oct27['label'].values
    for z in range(len(oct27)):
        feature = [int(i.split(":")[0]) for i in str(oct27['feature'][z]).strip().split(" ")][1:]
        a = set(feature)
        feature = list(a)
        feature.sort()
        # if len(a) != len(feature):
        #     print i, feature
        #     print 'error'
        #     exit()
        list_oct27_feature.append(str(feature).replace(',', ":1").replace('[', '').replace(']', '') + ':1')
    write_file = open(svm_name, 'w')

    for i in range(len(oct27)):
        aa = str(list_label[i]) + ' ' + str(list_oct27_feature[i])
        write_file.write(aa + '\n')


def data_load(path):
    from sklearn.datasets import load_svmlight_file
    data = load_svmlight_file(path)
    return data[0], data[1]


def one_hot():
    enc = OneHotEncoder()
    enc.fit([[2, 3, 1], [1, 3, 7], [1, 3, 8]])
    print "enc.n_values_ is:", enc.n_values_
    print "enc.feature_indices_ is:", enc.feature_indices_
    print enc.transform([[0, 1, 1]]).toarray()


covert_csv_to_svm('data/train.all.csv', 'data/oct.svm')
# one-hot  沒有在同一個向量空間
X, y = data_load('data/oct.svm')

# X = X_d
# y = y_d
# xg_train = xgb.DMatrix(X[0:int(X.shape[0] * 5.0 / 18)], label=y[0:int(X.shape[0] * 5.0 / 18)])
# xg_test = xgb.DMatrix(X[int(X.shape[0] * 5.0 / 18):], label=y[int(X.shape[0] * 5.0 / 18):])
import random

list_random = range(int(X.shape[0] - 7708))
random.shuffle(list_random)
split_num = X.shape[0] - 7708 + 1
# xg_train = xgb.DMatrix(X[0:int(X.shape[0] - 7708)][list_random], label=y[0:int(X.shape[0] - 7708)][list_random])
xg_train = xgb.DMatrix(X[0:int(split_num * 13.0 / 18)], label=y[0:int(split_num * 13.0 / 18)])
xg_test = xgb.DMatrix(X[int(split_num * 13.0 / 18):split_num], label=y[int(split_num * 13.0 / 18):split_num])
xg_daily = xgb.DMatrix(X[split_num:], label=y[split_num:])
params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'eval_metric': 'merror',
    'gamma': 0.1,  # 0.2 is ok
    'max_depth': 8,
    'subsample': 0.7,
    'min_child_weight': 2.5,
    'eta': 0.07,
    'lambda': 50,
    'seed': 713,
    'nthread': 4,
    'num_class': 25,
    'silent': 1
}

watchlist = [(xg_train, 'train'), (xg_test, 'test'),
             (xg_daily, 'daily')]  # The early stopping is based on last set in the evallist
num = 20
model = xgb.train(params, xg_train, num_boost_round=num, evals=watchlist)
model.save_model('data/xgb_13.model')
