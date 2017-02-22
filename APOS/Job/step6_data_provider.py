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

    def sentences_dict(self):
        import os
        from util import read_dict
        sentence_dict_path = 'pickle/' + 'id_sentences.pkl'
        if os.path.exists(sentence_dict_path) is False:
            print sentence_dict_path, ' does not exit'
            exit()
        id_sentences = read_dict(sentence_dict_path)
        texts = id_sentences.values()
        dic_t = {}
        for t in texts:
            for z in t:
                if dic_t.has_key(z):
                    continue
                else:
                    dic_t.setdefault(z, len(dic_t.keys()))
        return dic_t, id_sentences

    def w2v_feature(self):
        train = pd.read_csv('data/resume_clean.csv')
        # get id list
        id_list = list(train['id'])
        dic_t, id_sentences = self.sentences_dict()
        m = MyWordToVec(id_sentences.values(), 10)
        model = m.train_word2vec(save=True)
        print len(model.vocab)
        w2v_f = np.zeros((len(id_sentences), (4 + 3 * 4) * m.num_features))

        for id_index, temp_id in enumerate(id_list):
            sentence = id_sentences[temp_id]
            for index, token in enumerate(sentence):
                w2v_f[id_index][index * 10:index * 10 + 10] = model[sentence[index - 8]]
                if index >= 15:
                    break
        print w2v_f[0]
        return w2v_f


class MyWordToVec:
    num_features = 10  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 2  # Context window size
    down_sampling = 1e-2
    sentence = []
    sentence_dict = {}

    def __init__(self, sentence, vector_size):
        self.num_features = vector_size
        self.sentence = sentence

    def train_word2vec(self, word2vec_path='pickle/word2vec.model', save=False):
        from gensim.models.word2vec import Word2Vec
        import os
        if os.path.exists(word2vec_path):
            return Word2Vec.load(word2vec_path)
        else:
            print "Training Word2Vec model..."
            print len(self.sentence)
            model = Word2Vec(self.sentence, workers=self.num_workers, \
                             size=self.num_features, min_count=self.min_word_count, \
                             window=self.context, sample=self.down_sampling, seed=1, negative=1)
            if save is True:
                model.save(word2vec_path)
            print 'end training'
            return model


class MatrixAndDict:
    vec_size = 10
    dict_matrix = np.zeros((1, 1))

    def __init__(self, vector_size):
        self.vec_size = vector_size
        dic_t, id_sentences_dict = DataProvider().sentences_dict()
        model = MyWordToVec(id_sentences_dict.values(), self.vec_size).train_word2vec()
        self.dict_matrix = np.zeros((len(model.vocab), self.vec_size))
        keys = model.vocab.keys()
        for i in range(len(model.vocab)):
            self.dict_matrix[dic_t[keys[i]], :] = model[keys[i]]


"""

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

len(m.id_sentences_dict)
print m.dict_matrix.shape

"""
