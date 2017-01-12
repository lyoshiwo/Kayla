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

train = pd.read_csv('data/resume_clean.csv')
data = util.read_json('data/resume_clean.json')

[x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
count = 0
for i in range(x.shape[1]):
    count += x[:, i].max() + 1
    x[:, i] += count

# enc = preprocessing.OneHotEncoder()
# enc.fit(x)
# x = enc.transform(x)
y_ = np_utils.to_categorical(y, len(set(y)))
X_train, X_test, Y_train, Y_test = train_test_split(x, y_, test_size=0.25, random_state=713)


# X_train, X_validate, Y_train, Y_validate = train_test_split(x, y_, test_size=0.33, random_state=713)
# Y_tr, Y_te = train_test_split(y, test_size=0.33, random_state=713)


#
#
# # 27 epoch 0.461, 10 dim
# 18 epoch 0.4628, 20 dim two level
# 18 epoch 0.4653, 20 dim three level 0.5 dropout
# trainable=False 0.42

def mlp0():
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



def mlp1():
    X_train, X_validate, Y_train, Y_validate = train_test_split(x, y_, test_size=0.33, random_state=713)
    Y_tr, Y_te = train_test_split(y, test_size=0.33, random_state=713)
    # from keras.regularizers import l1
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
    model.add(Dense(input_dim=x.shape[1], output_dim=len(set(y))))
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

def mlp3():
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
    # model.add(Convolution1D(nb_filter=8,
    #                         filter_length=2,
    #                         border_mode='valid',
    #                         activation='relu'))
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


# mlp3()

# the first, the second, the last position
def sentence_to_matrix_vec(sentence, model, featuresNum, k_mean_dict_1, k_mean_dict_2, cluster=False, w2v=True):
    temp = np.zeros((featuresNum + 2) * (4 * 3 + 3))
    if sentence == None: return temp
    for i in range(-7, 2 * 4):
        try:
            if w2v is True:
                temp[(i + 7) * featuresNum:(i + 8) * featuresNum] = model[sentence[i]]
            if cluster is True:
                temp[15 * featuresNum + (i + 7) * 2] = k_mean_dict_1[sentence[i]]
                temp[15 * featuresNum + (i + 7) * 2 + 1] = k_mean_dict_2[sentence[i]]
        except Exception, e:
            print e
            continue
    return temp


def get_cluster_or_w2v_feature(data=data, cluster=False, w2v=True):
    from gensim.models import Word2Vec
    cluster_one_64 = util.read_dict("pickle/cluster_one_128.pkl")
    cluster_two_64 = util.read_dict("pickle/cluster_two_128.pkl")
    sentence_dict_path = 'pickle/id_sentences.pkl'
    word2vec_path = 'pickle/' + str(10) + 'features_1minwords_' + str(14) + 'context.pkl'
    sentence_dic = util.read_dict(sentence_dict_path)
    model = Word2Vec.load(word2vec_path)
    features = []
    sentences = [sentence_dic[i['id']] for i in data]
    for sentence in sentences:
        feature = sentence_to_matrix_vec(sentence, model, 10, cluster_one_64, cluster_two_64, cluster=cluster, w2v=w2v)
        features.append(feature)
    return features


def unsupervised_test(cluster=False, w2v=True):
    w2v_feature = get_cluster_or_w2v_feature(cluster=cluster, w2v=w2v)
    x = w2v_feature
    x = np.matrix(x)
    print x.shape
    X_train, X_test, Y_train, Y_test = train_test_split(x, y_, test_size=0.33, random_state=713)
    model = Sequential()
    model.add(Dense(128, input_dim=x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5
                      ))
    model.add(Activation('relu'))
    model.add(Dense(len(set(y))))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    batch_size = 1024
    nb_epoch = 20
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch * 50,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def unsupervised_test_cnn(cluster=False, w2v=True):
    w2v_feature = get_cluster_or_w2v_feature(cluster=cluster, w2v=w2v)
    x = np.array(w2v_feature)
    print x.shape
    x = x.reshape((x.shape[0], 1, 10, 18))
    X_train, X_test, Y_train, Y_test = train_test_split(x, y_, test_size=0.33, random_state=713)
    model = Sequential()
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    model.add(Convolution2D(32, 3, 3, input_shape=X_train.shape[1:]))
    # model.add(Convolution1D(nb_filter=8,
    #                         filter_length=2,
    #                         border_mode='valid',
    #                         activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(len(set(y))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=1024, nb_epoch=20 * 15,
              validation_data=(X_test, Y_test))
    import time
    print time.localtime()


# unsupervised_test_cnn()

def print_img():
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

    # vec = get_cluster_or_w2v_feature(data)
    #
    # for index, v in enumerate(vec):
    #     if y[index] == 0 or y[index] == 5:
    #         for i, k in enumerate(v.reshape((18, 10))):
    #             if i != 4:
    #                 continue
    #             new_x.append(k)
    #             new_y.append(y[index])
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
