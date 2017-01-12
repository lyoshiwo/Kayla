#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, grid_search
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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


# 0.4238 1000
def test_rf(X_train, Y_train, X_test, Y_test):
    clf = RandomForestClassifier(n_estimators=600)
    clf.fit(X_train, Y_train)
    y_predict = clf.predict(X_test)
    evaluation(Y_test, y_predict)


def test_tf_cv(X_train, Y_train):
    print 'rf'
    n_estimator = range(100, 301, 100)
    max_depth = range(5, 26, 1)
    clf = RandomForestClassifier(n_jobs=4)
    parameters = {'n_estimators': n_estimator, 'max_depth': max_depth}
    grid_clf = grid_search.GridSearchCV(clf, parameters)
    grid_clf.fit(np.array(X_train), np.array(Y_train))
    score = grid_clf.grid_scores_
    l1 = [1 - x[1] for x in score if x[0]['n_estimators'] == n_estimator[0]]
    l2 = [1 - x[1] for x in score if x[0]['n_estimators'] == n_estimator[1]]
    l3 = [1 - x[1] for x in score if x[0]['n_estimators'] == n_estimator[2]]
    plt.plot(range(5, 26, 1), l1,
             'b--')
    plt.plot(range(5, 26, 1), l2,
             'r.--')
    plt.plot(range(5, 26, 1), l3,
             'g')
    plt.legend((str(n_estimator[0]) + ' estimators', str(n_estimator[1]) + ' estimators',
                str(n_estimator[2]) + ' estimators'),
               loc=0, shadow=True)
    plt.xlabel('max depth of RandomForest')
    plt.ylabel('average error rate of  3-fold cross-validation')
    plt.grid(True)
    plt.show()


def xgb_test_cv(X_train, Y_train):
    from pandas import Series
    import xgboost as xgb
    from util import write_file,read_file
    import os
    if os.path.exists('pkl/history,pkl') is False:
        set_y = set(Y_train)
        param["num_class"] = len(set_y)
        dtrain = xgb.DMatrix(np.array(X_train), label=list(Y_train))
        print "this is cv func"
        param['max_depth'] = 4
        history4 = xgb.cv(param, dtrain, 500, nfold=3, verbose_eval=True)
        param['max_depth'] = 6
        history6 = xgb.cv(param, dtrain, 500, nfold=3, verbose_eval=True)
        param['max_depth'] = 8
        history8 = xgb.cv(param, dtrain, 500, nfold=3, verbose_eval=True)
        write_file([history4, history6, history8], 'pickle/history,pkl')
    else:
        [history4, history6, history8]=read_file('pickle/history,pkl')
    test_error_list = Series(history4['test-merror-mean']).values[50:]
    plt.plot(range(51, len(test_error_list) + 1), test_error_list, 'b--')
    test_error_list = Series(history6['test-merror-mean']).values[50:]
    plt.plot(range(51, len(test_error_list) + 1), test_error_list, 'r:')
    test_error_list = Series(history8['test-merror-mean']).values[50:]
    plt.plot(range(51, len(test_error_list) + 1), test_error_list, 'g')
    plt.ylabel('average error rate of  3-fold cross-validation')
    plt.legend(('max depth of xgb is 4', 'max depth of xgb is 6', 'max depth of xgb is 8'),
               loc=0, shadow=True)
    plt.xlabel('the number of boost round')
    plt.show()

# test_rf()

# 400 0.454 70-features
def test_xgb(X_train, Y_train, X_test, Y_test, y):
    import xgboost as xgb
    import time
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    set_y = set(y)
    param["num_class"] = len(set_y)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)
    param['objective'] = 'multi:softmax'
    xgb_model = xgb.train(param, dtrain, num_boost_round=600)
    y_predict = xgb_model.predict(dtest)
    evaluation(Y_test, y_predict)

    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def evaluation(y_test, y_prediction):
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    print 'precision: ', precision_score(y_test, y_prediction, average='micro')
    print 'recall: ', recall_score(y_test, y_prediction, average='micro')
    print 'f1 score: ', f1_score(y_test, y_prediction, average='micro')


if __name__ == "__main__":
    from step5_data_provider import DataProvider

    d = DataProvider(random_state=713, test_size=0.33)
    d.manuel_feature()
    for name in ['position', 'salary', 'size']:
        print name
        X_train = d.x_train_test['train']
        Y_train = d.y[name]['train']
        X_test = d.x_train_test['test']
        Y_test = d.y[name]['test']
        y = d.y_original[name]
        xgb_test_cv(X_train, Y_train)
        test_xgb(X_train, Y_train, X_test, Y_test, y)
        test_rf(X_train, Y_train, X_test, Y_test)
