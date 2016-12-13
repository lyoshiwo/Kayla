#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

[x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=713)
enc = preprocessing.OneHotEncoder()
enc.fit(x)
x = enc.transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=713)


# 0.4238 1000
def test_rf():
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_train, Y_train)
    y_test = clf.predict(X_test)
    print metrics.accuracy_score(y_test, Y_test)


# test_rf()

# 651 0.454 70-features
def test_xgb(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test):
    param = dict()
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.03
    param['max_depth'] = 8
    param['eval_metric'] = 'merror'
    param['silent'] = 1
    param['min_child_weight'] = 10
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.2
    param['nthread'] = 3
    param['num_class'] = -1
    import xgboost as xgb
    import time
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    set_y = set(y)
    param["num_class"] = len(set_y)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)
    param['objective'] = 'multi:softmax'
    xgb.train(param, dtrain, num_boost_round=1000, evals=[(dtrain, 'train'), (dtest, 'validate')])
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


test_xgb()
