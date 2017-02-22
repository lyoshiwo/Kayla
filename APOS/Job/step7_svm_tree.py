#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from matplotlib import pyplot as plt
import numpy as np

param = dict()
param['objective'] = 'multi:softmax'
param['eta'] = 0.03
param['max_depth'] = 6
param['eval_metric'] = 'merror'
param['silent'] = 1
param['min_child_weight'] = 10
param['subsample'] = 0.7
param['colsample_bytree'] = 0.2
param['nthread'] = 3
param['num_class'] = -1


# 0.4238 1000
def test_rf(X_train, Y_train, X_test, Y_test, n_estimators=600):
    print "test_rf"
    print 'n_estimators', n_estimators
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, Y_train)
    y_predict = clf.predict(X_test)
    evaluation(Y_test, y_predict)


def test_tf_cv(X_train, Y_train):
    print 'cv_rf'
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
    from util import write_file, read_file
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
        [history4, history6, history8] = read_file('pickle/history,pkl')
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
def test_xgb(X_train, Y_train, X_test, Y_test, y, name, boost_dict):
    num_boost_round = boost_dict[name]
    print 'test xgb'
    import xgboost as xgb
    import time
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    set_y = set(y)
    param["num_class"] = len(set_y)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)
    param['objective'] = 'multi:softmax'
    # watchlist = [(dtest, 'test')]
    # xgb_model = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=watchlist)
    xgb_model = xgb.train(param, dtrain, num_boost_round=num_boost_round)
    y_predict = xgb_model.predict(dtest)
    evaluation(Y_test, y_predict)

    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def evaluation(y_test, y_prediction):
    from sklearn.metrics import precision_score
    print 'precision: ', precision_score(y_test, y_prediction, average='micro')
    print 'precision: ', precision_score(y_test, y_prediction, average='macro')


def test_svm(X_train, Y_train, X_test, Y_test):
    from sklearn.svm import LinearSVC
    clf = LinearSVC(penalty='l1', dual=False, max_iter=3000, verbose=1)
    clf.fit(X_train, Y_train)
    Y_prediction = clf.predict(X_test)
    evaluation(Y_test, Y_prediction)


if __name__ == "__main__":
    from step6_data_provider import DataProvider
    from sklearn.cross_validation import train_test_split

    d = DataProvider(random_state=713, test_size=0.33)
    X_train = d.x_train_test['train']
    X_test = d.x_train_test['test']
    X_hot_train, X_hot_test = d.one_hot()
    X_embedding = d.w2v_feature()
    X_embedding_train, X_embedding_test = train_test_split(X_embedding, test_size=d.test_size,
                                                           random_state=d.random_state)
    print X_train.shape, X_embedding_train.shape
    # useless
    # X_train = np.concatenate([X_train, X_embedding_train], axis=1)
    # X_test = np.concatenate([X_test, X_embedding_test], axis=1)
    # X_hot_train = np.concatenate([X_hot_train, X_embedding_train], axis=1)
    # X_hot_test = np.concatenate([X_hot_test, X_embedding_test], axis=1)
    boost_dict = {'size': 228, 'salary': 561, 'position': 380}
    for name in ['position']:
        print "*******************"
        print name
        Y_train = d.y[name]['train']
        Y_test = d.y[name]['test']
        y = d.y_original[name]
        xgb_test_cv(X_train, Y_train)
        # test_xgb(X_train, Y_train, X_test, Y_test, y, name, boost_dict)
        # test_rf(X_train, Y_train, X_test, Y_test)
        # test_svm(X_hot_train, Y_train, X_hot_test, Y_test)
