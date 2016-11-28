#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import pandas as pd
import numpy as np
import xgboost as xgb
import math


def data_load(path):
    from sklearn.datasets import load_svmlight_file
    data = load_svmlight_file(path)
    return data[0], data[1]


def get_soft_max(scores):
    score_list = []
    for score in scores:
        z = max(score)
        index = list(score).index(z)
        score_list.append(index)
    return score_list


X, y = data_load('data/oct.svm')
s = pd.Series(y, name='label')
matrix = np.zeros([len(s.value_counts()), len(s.value_counts())])
for i in range(len(y) - 1):
    matrix[y[i]][y[i + 1]] += 1
for i in range(matrix.shape[0]):
    matrix[i] += 0.00000001 / 25
    matrix[i] /= sum(matrix[i])

model = xgb.Booster({'nthread': 4})  # init model
model.load_model('data/xgb_13.model')  # load data
split_num = X.shape[0] - 7708 + 1
xg_daily = xgb.DMatrix(X[split_num:], label=y[split_num:])
scores = model.predict(xg_daily)
sentences = []
sentence = []

oct27 = pd.read_csv('data/train.all.csv')
list_oct27_feature = []
list_label = oct27['label'].values
edge = []
for k in range(len(oct27)):
    feature = [int(i.split(":")[0]) for i in str(oct27['feature'][k]).strip().split(" ")][0]
    edge.append(feature)
for i in range(len(scores)):
    if edge[split_num:][i] == 25 and i != 0:
        sentences.append(sentence)
        sentence = [scores[i]]
    else:
        sentence.append(scores[i])
edge = edge[split_num:]

print y[split_num:][:30]
print get_soft_max(sentences[0])


# print get_soft_max(sentences[1])


def viterbi(sentences, matrix):
    num = 0
    for score in sentences:
        temp_matrix = np.zeros([matrix.shape[0], matrix.shape[1]])
        path_matrix = np.zeros([matrix.shape[0], len(score) - 1])
        max_vec = np.array(score[0])
        max_vec += 0.0001
        max_vec = np.log(max_vec)
        import copy
        temp_max_vec = copy.deepcopy(max_vec)
        for i in range(len(score) - 1):
            now_vec = np.zeros([matrix.shape[0]]) - float('Inf')
            for col in range(matrix.shape[1]):
                for row in range(matrix.shape[0]):
                    # multiply 2 in case the value too small
                    temp_matrix[row, col] = math.log(matrix[row, col]) + max_vec[row] + math.log(
                        score[i + 1][col] + 0.0001)
                    if temp_matrix[row, col] > now_vec[col]:
                        now_vec[col] = temp_matrix[row, col]
                        path_matrix[col, i] = row
                temp_max_vec[col] = now_vec[col]
            max_vec = copy.deepcopy(temp_max_vec)
        kkk = get_soft_max([max_vec])
        final = kkk[0]
        for i in range(len(score) - 1):
            kkk.append(path_matrix[int(final), int(len(score) - 2 - i)])
            final = path_matrix[int(final), int(len(score) - 2 - i)]
        kkk.reverse()
        kkk = [int(i) for i in kkk]
        print kkk


# for row in range(matrix.shape[0]):
#     for col in range(matrix.shape[1]):
#         print row, col, matrix[row, col]
# exit()

# sentences = [[[0.3, 0.2, 0.5], [0.6, 0.2, 0.2]], [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]]]
sentences = [sentences[0][9:]]
# print [int(i) for i in y[split_num:][:len(sentences[0])]]
viterbi(sentences, matrix)
matrix = np.ones([matrix.shape[0], matrix.shape[1]])
viterbi(sentences, matrix)
num = 0
z = [3, 9, 3, 5, 6, 4, 7]
zz = [3, 9, 4, 5, 18, 4, 1]
for i in sentences:
    print i
print len(sentences[0])
