#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.cross_validation import train_test_split
from keras.layers.embeddings import Embedding
from sklearn import preprocessing
[x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
# enc = preprocessing.OneHotEncoder()
# enc.fit(x)
# x = enc.transform(x)
y_ = np_utils.to_categorical(y, len(set(y)))
X_train, X_test, Y_train, Y_test = train_test_split(x, y_, test_size=0.33, random_state=713)


# 38 epoch,0.456
def mlp():
    # from keras.regularizers import l1
    model = Sequential()
    model.add(Embedding(x.max() + 2, 10, input_length=x.shape[1]))
    model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(y))))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    batch_size = 1024
    nb_epoch = 1000
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# mlp()


def mlp2():
    [x, _, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
    enc = preprocessing.OneHotEncoder()
    enc.fit(x)
    feature_num = len(enc.active_features_)
    one_hot_x = enc.transform(x)
    indexs = one_hot_x.indices
    blocks = one_hot_x.indptr
    print len(indexs)
    print len(blocks)
    # all_data = np.zeros((x.shape[0],feature_num))


mlp2()
