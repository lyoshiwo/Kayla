#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import numpy as np


# give categorical data
class DataProvider:
    random_state = 713
    y = {'position': {}, 'size': {}, 'salary': {}}
    y['position'] = {'train': [], 'test': []}
    y['size'] = {'train': [], 'test': []}
    y['salary'] = {'train': [], 'test': []}
    x_train_test = {'train': [], 'test': []}
    x_original = []
    y_original = {'position': [], 'size': [], 'salary': []}
    test_size = 0.33

    def __init__(self, random_state=713, test_size=0.33):
        self.test_size = test_size
        self.random_state = random_state

    def manuel_feature(self):
        [self.x_original, position, size, salary] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
        x_original = self.x_original
        self.x_train_test['train'], self.x_train_test['test'] = train_test_split(x_original, test_size=self.test_size,
                                                                                 random_state=self.random_state)
        self.y['position']['train'], self.y['position']['test'] = train_test_split(position, test_size=self.test_size,
                                                                                   random_state=self.random_state)
        self.y['size']['train'], self.y['size']['test'] = train_test_split(size, test_size=self.test_size,
                                                                           random_state=self.random_state)
        self.y['salary']['train'], self.y['salary']['test'] = train_test_split(salary, test_size=self.test_size,
                                                                               random_state=self.random_state)
        self.y_original = {'position': position, 'size': size, 'salary': salary}

    def one_hot(self):
        """
        :return: one_hot_train_list, one_hot_test_list
        """
        count = 0
        import numpy as np
        self.x_original = np.array(self.x_original)
        for i in range(self.x_original.shape[1]):
            count += self.x_original[:, i].max() + 1
            self.x_original[:, i] += count
        enc = preprocessing.OneHotEncoder()
        enc.fit(self.x_original)
        self.x_original = enc.transform(self.x_original)
        return train_test_split(self.x_original, test_size=self.test_size, random_state=self.random_state)

    def text_dict(self):
        import os
        from util import read_dict
        """
        :return: matrix and dict with key is token, value is index
        """
        sentence_dict_path = 'pickle/' + 'id_sentences.pkl'
        if os.path.exists(sentence_dict_path) is False:
            print sentence_dict_path, ' does not exit'
            exit()
        id_text = read_dict(sentence_dict_path)
        texts = id_text.values()
        dic_t = {}
        for t in texts:
            for z in t:
                if dic_t.has_key(z):
                    continue
                else:
                    dic_t.setdefault(z, len(dic_t.keys()))
        return texts, dic_t, id_text


class MyWordToVec:
    num_features = 10  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 2  # Context window size
    down_sampling = 1e-3
    sentence = []
    sentence_dict = {}

    def __init__(self, sentence, vector_size):
        self.num_features = vector_size
        self.sentence = sentence

    def train_word2vec(self):
        from gensim.models.word2vec import Word2Vec
        word2vec_path = 'pickle/word2vec.model'
        import os
        if os.path.exists(word2vec_path):
            return Word2Vec.load(word2vec_path)
        else:
            print "Training Word2Vec model..."
            model = Word2Vec(self.sentence, workers=self.num_workers, \
                             size=self.num_features, min_count=self.min_word_count, \
                             window=self.context, sample=self.down_sampling, seed=1, negative=1)
            model.save(word2vec_path)
            return model


class MatrixAndDict:
    vec_size = 10
    text, dic_t, id_text_dict = DataProvider().text_dict()
    model = MyWordToVec(text, vec_size).train_word2vec()
    dict_matrix = np.zeros((len(model.vocab), vec_size))

    def __init__(self, vector_size):
        self.vec_size = vector_size
        keys = self.model.vocab.keys()
        for i in range(len(self.model.vocab)):
            self.dict_matrix[self.dic_t[keys[i]], :] = self.model[keys[i]]


# demo
train = pd.read_csv('data/resume_clean.csv')
# get id list
print len(train['id'])
# define random state and vector size
random_state = 713
test_size = 0.33
vec_size = 10
# get word embedding with token id
m = MatrixAndDict(vec_size)
# get manual feature
d = DataProvider(random_state, test_size)
d.manuel_feature()

len(m.id_text_dict)
print m.dict_matrix.shape
