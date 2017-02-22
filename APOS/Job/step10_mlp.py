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

def mlp3(x, y, X_train, Y_train, X_test, Y_test):
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
    sentence_small = sentence_dic.values()
    from step6_data_provider import MyWordToVec
    m = MyWordToVec(sentence_small, 10)
    model = m.train_word2vec("", False)
    print model.most_similar(u"销售经理", topn=10)
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


def show_validate():
    import matplotlib.pyplot as plt
    l1 = [0.2307, 0.3760, 0.4214, 0.4470, 0.4642, 0.4733, 0.4872, 0.4922, 0.5005, 0.5110, 0.5154, 0.5231, 0.5279,
          0.5388, 0.5469]
    l2 = [0.3702, 0.4243, 0.4453, 0.4543, 0.4575, 0.4639, 0.4623, 0.4660, 0.4657, 0.4645, 0.4631, 0.4642, 0.4623,
          0.4609, 0.4576]
    l3 = [0.2107, 0.3318, 0.3834, 0.4128, 0.4313, 0.4495, 0.4629, 0.4715, 0.4837, 0.4927, 0.5026, 0.5083, 0.5193,
          0.5292, 0.5347]
    l4 = [0.3295, 0.3986, 0.4283, 0.44418, 0.4505, 0.4568, 0.4595, 0.4624, 0.4654, 0.4640, 0.4657, 0.4649, 0.4625,
          0.4580, 0.4567]
    plt.plot(range(1, len(l1) + 1, 1), l1,
             'b.--')
    plt.plot(range(1, len(l1) + 1, 1), l3,
             'r--')
    plt.plot(range(1, len(l1) + 1, 1), l2,
             'g.-')
    plt.plot(range(1, len(l1) + 1, 1), l4,
             'y')
    plt.legend(("final DNN: train accuracy rate","original DNN: train accuracy rate", "final DNN: validation accuracy rate","original DNN: validation accuracy rate"),
               loc=0, shadow=True)
    plt.xlabel('epoch of DNN models')
    plt.ylabel('accuracy rate of position prediction')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    show_validate()
    from step6_data_provider import DataProvider, MatrixAndDict

    [x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
    vec_size = 10
    m = MatrixAndDict(vec_size)
    d = DataProvider(random_state=713, test_size=0.33)
    print len(d.y_original["salary"])
    # print_img(d.y_original["salary"])
    dict_token, id_text_dict = d.sentences_dict()
    X_train = d.x_train_test['train']
    X_test = d.x_train_test['test']
    train = pd.read_csv('data/resume_clean.csv')
    id_list = train['id']
    index_list = []
    my_matrix = m.dict_matrix
    # every resume has 2 jobs at least; keep tree jobs, the last , the last but second, and the first;
    for id in id_list:
        text = id_text_dict[id]
        temp = [dict_token[t] for t in text]
        index_list.append(temp[-8:] + temp[0:8])
    from sklearn.cross_validation import train_test_split

    index_list_train, index_list_test = train_test_split(index_list, random_state=713, test_size=0.33)

    import time

    print time.localtime()
    import theano
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import np_utils

    target = 'position'
    data_train = pad_sequences(index_list_train, maxlen=16)
    data_test = pad_sequences(index_list_test, maxlen=16)
    y_train = np_utils.to_categorical(np.asarray(d.y[target]['train']))
    y_test = np_utils.to_categorical(np.asarray(d.y[target]['test']))
    model1 = Sequential()
    model1.add(Embedding(x.max() + 1, 30, input_length=x.shape[1]))
    model1.add(Flatten())
    # model1.add(Dense())
    # model1.add(Activation('tanh'))


    model2 = Sequential()
    model2.add(Embedding(np.array(dict_token.values()).max() + 1, my_matrix.shape[1], weights=[my_matrix],
                         input_length=16,
                         trainable=True))
    model2.add(Flatten())
    model2.add(Dense(1024))
    model2.add(Activation('relu'))
    from keras.layers.core import Merge

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=1))
    # model = model1
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(d.y_original[target]))))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    batch_size = 1024
    nb_epoch = 50
    model.fit([np.array(X_train), np.array(index_list_train)], y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=([np.array(X_test), np.array(index_list_test)], y_test))
    # model.fit(np.array(X_train), y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #           validation_data=(np.array(X_test), y_test))
    model.summary()
