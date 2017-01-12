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
from gensim.models import Word2Vec
import numpy as np
import util


# # 27 epoch 0.461, 10 dim
# 18 epoch 0.4628, 20 dim two level
# 18 epoch 0.4653, 20 dim three level 0.5 dropout
# trainable=False 0.42

def mlp0(y, x, X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(len(set(y)), input_dim=x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(y))))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    batch_size = 1024
    nb_epoch = 20
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# mlp0()



def mlp1(y, x, X_train, Y_train, X_test, Y_test):

    import theano
    print x.max() + 1
    model = Sequential()
    model.add(Embedding(x.max() + 1, 30, input_length=x.shape[1]))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(y))))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    batch_size = 1024
    nb_epoch = 25
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))
    model.summary()
    exit()
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    get_feature = theano.function([model.layers[0].input], model.layers[0].output, allow_input_downcast=True)
    new_go = get_feature(X_train)
    new_train = new_go.reshape([new_go.shape[0], new_go.shape[1] * new_go.shape[2]])
    new_test = get_feature(X_test)
    new_test = new_test.reshape([new_test.shape[0], new_test.shape[1] * new_test.shape[2]])
    print 'end'

# mlp1()

def mlp3(x,y,X_train, Y_train,X_test, Y_test):
    import time
    print time.localtime()
    import theano
    model = Sequential()
    model.add(Embedding(x.max() + 1, 30, input_length=x.shape[1]))
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
    nb_epoch = 12
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    get_feature = theano.function([model.layers[0].input], model.layers[0].output, allow_input_downcast=True)
    new_go = get_feature(X_train)
    new_train = new_go.reshape((new_go.shape[0], 1, new_go.shape[1], new_go.shape[2]))
    print new_train.shape
    new_test = get_feature(X_test)
    new_test = new_test.reshape((new_test.shape[0], 1, new_test.shape[1], new_test.shape[2]))
    print new_test.shape
    # build a 2D model
    from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
    print 'building model'
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=new_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(len(set(y))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit(new_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch * 5,
              validation_data=(new_test, Y_test))
    print time.localtime()

    print 'saved model'
    json_string = model.to_json()
    import util
    util.write_file(json_string, 'pickle/json_string_2d_cnn.pkl')
    model.save_weights('pickle/weight_2d_cnn.h5')

def print_img(y):
    sentence_dict_path = 'pickle/id_sentences.pkl'
    sentence_dic = util.read_dict(sentence_dict_path)
    print sentence_dic.values()[0]
    sentence_small = []
    for s in sentence_dic.values():
        sentence_small.append([s[i] for i in range(len(s)) if
                               i % 4 == 2 or i + 3 == len(s)][:-1])

    num_features = 10  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 1  # Context window size
    downsampling = 1e-2  # Downsample setting for frequent words

    print "Training Word2Vec model..."
    model = Word2Vec(sentence_small, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1, negative=-1)
    print "print img"

    new_x = []
    new_y = []
    for index, s in enumerate(sentence_small[:5000]):
        if y[index] % 6 != 0:
            continue
        for v in s:
            try:
                vv = model[v]
                new_x.append(vv)
                new_y.append(y[index])
            except:
                continue
    print len(new_x)
    print len(new_y)
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(122)
    import numpy as np
    new_x = np.matrix(new_x)
    X_pca = PCA().fit_transform(new_x)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=new_y)

    from sklearn.manifold import TSNE
    X_tsne = TSNE(learning_rate=100).fit_transform(new_x)
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=new_y)

    plt.show()


print_img()
