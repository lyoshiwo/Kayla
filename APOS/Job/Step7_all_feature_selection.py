# encoding=utf8
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys
from sklearn_pandas import DataFrameMapper
import util
import pandas as pd

train = pd.read_csv('data/resume_clean.csv')
data = util.read_json('data/resume_clean.json')


def sentence_to_matrix_vec(sentence, model, featuresNum, k_mean_dict_1, k_mean_dict_2):
    temp = np.zeros((featuresNum * (7 * 5 + 3) + 7 * 5 * 2))
    if sentence == None: return temp
    num = (len(sentence) - 3) / 7 if (len(sentence) - 3) / 7 <= 5 else 5
    for i in range(num * 7):
        temp[featuresNum * i:featuresNum * (i + 1)] = model[sentence[i]]
        try:
            temp[38 * featuresNum + i * 2] = k_mean_dict_1[sentence[i]]
            temp[38 * featuresNum + i * 2 + 1] = k_mean_dict_2[sentence[i]]
        except Exception, e:
            continue
    for i in range(3):
        temp[(5 * 7 + i) * featuresNum:(5 * 7 + i + 1) * featuresNum] = model[sentence[-1 * (i + 1)]]
    return temp


def get_manual_feature(train):
    print len(train.keys())
    mapper = util.get_mapper(train)
    train = mapper.transform(train)
    print train.shape
    return train


def get_cluster_feature(data):
    cluster_one_64 = util.read_dict("pickle/cluster_two_128.pkl")
    cluster_two_64 = util.read_dict("pickle/cluster_one_128.pkl")
    # print cluster_one_64.keys()
    # print 'aa'
    # sentence_dict_path = 'pickle/id_sentences.pkl'
    # word2vec_path = 'pickle/' + str(10) + 'features_1minwords_' + str(14) + 'context.pkl'
    # sentence_dic = util.read_dict(sentence_dict_path)
    # sentences = sentence_dic.values()
    # for i in sentences:
    #     for z in i:
    #         if cluster_one_64.has_key(z):
    #             print cluster_one_64[z]
    # exit()

    sentence_dict_path = 'pickle/id_sentences.pkl'
    word2vec_path = 'pickle/' + str(10) + 'features_1minwords_' + str(14) + 'context.pkl'
    sentence_dic = util.read_dict(sentence_dict_path)
    model = Word2Vec.load(word2vec_path)
    features = []
    sentences = [sentence_dic[i['id']] for i in data]
    for sentence in sentences:
        feature = sentence_to_matrix_vec(sentence, model, 10, cluster_one_64, cluster_two_64)
        features.append(feature)
    return features


features = get_cluster_feature(data)
for i in range(100):
    print features[i][380:]
