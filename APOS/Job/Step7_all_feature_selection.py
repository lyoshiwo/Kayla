# encoding=utf8
from gensim.models import Word2Vec
import numpy as np
import util
import pandas as pd

train = pd.read_csv('data/resume_clean.csv')
data = util.read_json('data/resume_clean.json')


def sentence_to_matrix_vec(sentence, model, featuresNum, k_mean_dict_1, k_mean_dict_2):
    temp = np.zeros((featuresNum + 2) * (4 * 5 + 3))
    if sentence == None: return temp
    num = (len(sentence) - 3) / 4 if (len(sentence) - 3) / 4 <= 5 else 5
    for i in range(-3, num * 4):
        try:
            temp[(i + 3) * featuresNum:(i + 4) * featuresNum] = model[sentence[i]]
            temp[23 * featuresNum + (i + 3) * 2] = k_mean_dict_1[sentence[i]]
            temp[23 * featuresNum + (i + 3) * 2 + 1] = k_mean_dict_2[sentence[i]]
        except Exception, e:
            continue
    return temp


def get_manual_feature(train):
    print len(train.keys())
    mapper = util.get_mapper(train)
    train = mapper.transform(train)
    print train.shape
    return train


def get_cluster_feature(data):
    cluster_one_64 = util.read_dict("pickle/cluster_one_128.pkl")
    cluster_two_64 = util.read_dict("pickle/cluster_two_128.pkl")
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
print len(features)
[x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
print len(x), len(y)


# 230 0.552-error
# all 0.554-error
def test_xgb(x, y):
    from sklearn import preprocessing
    from sklearn.cross_validation import train_test_split
    # enc = preprocessing.OneHotEncoder()
    # enc.fit(x)
    # x = enc.transform(x)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=713)

    param = dict()
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['eval_metric'] = 'merror'
    param['silent'] = 1
    param['min_child_weight'] = 10
    param['subsample'] = 0.4
    param['colsample_bytree'] = 0.2
    param['nthread'] = 2
    param['num_class'] = -1
    import xgboost as xgb
    import time

    train_Y = y
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    set_y = set(train_Y)
    param["num_class"] = len(set_y)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)
    param['objective'] = 'multi:softmax'
    xgb.train(param, dtrain, num_boost_round=1000, evals=[(dtrain, 'train'), (dtest, 'validate')])
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


now_max = x.max()
features = [np.array(i[230:]) for i in features]
features = np.concatenate((features, x), axis=1)
print features.shape


# 346 0.5447-error; 300 0.5457; 116 0.5470
# test_xgb(features, y)

# 116 0.4650
def mlp(features, y):
    features = np.matrix(features)
    print features.shape
    import pandas as pd
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.optimizers import SGD, Adam, RMSprop
    from sklearn.cross_validation import train_test_split
    from keras.layers.embeddings import Embedding
    # from keras.regularizers import l1
    model = Sequential()
    model.add(Embedding(features.max() + 1, 20, input_length=features.shape[1]))
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
    y = np_utils.to_categorical(y, len(set(y)))
    X_train, X_test, Y_train, Y_test = train_test_split(features, y, test_size=0.33, random_state=713)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


mlp(features, y)
