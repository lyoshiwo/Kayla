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
import numpy as np

[x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
count = 0
for i in range(x.shape[1]):
    count += x[:, i].max() + 1
    x[:, i] += count

# enc = preprocessing.OneHotEncoder()
# enc.fit(x)
# x = enc.transform(x)
y_ = np_utils.to_categorical(y, len(set(y)))
X_train, X_test, Y_train, Y_test = train_test_split(x, y_, test_size=0.33, random_state=713)


#
#
# # 27 epoch 0.461, 10 dim
# 18 epoch 0.4628, 20 dim two level
# 18 epoch 0.4653, 20 dim three level 0.5 dropout
# trainable=False 0.42
def mlp():
    # from keras.regularizers import l1
    model = Sequential()
    model.add(Embedding(x.max() + 1, 20, input_length=x.shape[1], trainable=False))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(y))))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    batch_size = 1024
    nb_epoch = 300
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


mlp()


def mlp2():
    from sklearn import preprocessing
    import pandas as pd
    import numpy as np
    [x, _, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
    enc = preprocessing.OneHotEncoder()
    enc.fit(x)
    one_hot_x = enc.transform(x)
    indexs = one_hot_x.indices
    blocks = one_hot_x.indptr
    print len(indexs)
    print len(blocks)
    all_data = np.zeros((one_hot_x.shape[0], one_hot_x.shape[1]))
    print blocks[-1]
    print indexs[blocks[-1]:]
    for b in range(one_hot_x.shape[0]):
        for v in indexs[blocks[b]:blocks[b + 1]]:
            all_data[b][v] = 1
    model = Sequential()
    x = all_data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y_, test_size=0.33, random_state=713)
    from keras.regularizers import l1
    model.add(Dense(input_dim=x.shape[0], output_dim=len(set(y)), W_regularizer=l1(0.01)))
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

# all_data = np.zeros((x.shape[0],feature_num))

# can not be initial, because the first level is too huge
# mlp2()
