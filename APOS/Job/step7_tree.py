#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

[x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=713)


# 0.421632
def test_rf():
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, Y_train)
    y_test = clf.predict(X_test)
    print metrics.accuracy_score(y_test, Y_test)


# test_rf()

# 200,0.44；
def test_xgb():
    from sklearn import preprocessing
    enc = preprocessing.OneHotEncoder()
    [x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
    enc.fit(x)
    x = enc.transform(x)
    # X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=713)

    param = dict()
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['eval_metric'] = 'merror'
    param['silent'] = 1
    param['min_child_weight'] = 10
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.2
    param['nthread'] = 2
    param['num_class'] = -1
    import xgboost as xgb
    import time

    train_Y = y
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    set_y = set(train_Y)
    param["num_class"] = len(set_y)
    dtrain = xgb.DMatrix(x, label=y)
    param['objective'] = 'multi:softmax'
    xgb.cv(param, dtrain, 1200, nfold=3, show_progress=True)
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


test_xgb()
