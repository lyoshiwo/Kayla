#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import metrics


# 0.40
def test_linearsvc():
    from sklearn import preprocessing, cross_validation
    [x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
    enc = preprocessing.OneHotEncoder()
    enc.fit(x)
    x = enc.transform(x)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.33, random_state=713)
    clf = LinearSVC(penalty='l1', dual=False, max_iter=1000, verbose=1)
    clf.fit(x_train, y_train)
    y1 = clf.predict(x_test)
    print metrics.accuracy_score(y1, y_test)


test_linearsvc()
