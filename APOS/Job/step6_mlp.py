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


# 0.45
def mlp():
    from keras.regularizers import l1
    model = Sequential()
    model.add(Embedding(x.max() + 2, 10, input_length=x.shape[1], W_regularizer=l1(0.01)))
    model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(len(set(y)), W_regularizer=l1(0.01)))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    batch_size = 1024
    nb_epoch = 100
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


mlp()