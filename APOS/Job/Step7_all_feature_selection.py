# encoding=utf8
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys
from sklearn_pandas import DataFrameMapper
import util
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf-8')

features = []
all_features = features + ["predict_degree", "predict_salary", "predict_size", "predict_position_name"]

train = pd.read_pickle(util.features_prefix + "manual_feature.pkl")
print len(train), len(features), len(all_features)
train = train[all_features]
train = train[train["predict_position_name"].isin(util.position_name_list)]
data_all = pd.concat([train[features]])


def get_mapper(data_all):

    return mapper


mapper = get_mapper(data_all)


def getPrecision(multiclf, train_X, train_Y, label_dict):
    pred_Y = multiclf.predict(train_X)
    pred_Y = [int(p) for p in pred_Y]

    print "total accuracy_score%s" % (accuracy_score(train_Y, pred_Y))
    diff_num = len(label_dict.classes_)

    for i in xrange(diff_num):
        hit, test_cnt, pred_cnt = 0, 0, 0
        for k in xrange(len(train_Y)):
            if train_Y[k] == i:
                test_cnt += 1
            if pred_Y[k] == i:
                pred_cnt += 1
            if train_Y[k] == i and pred_Y[k] == i:
                hit += 1
        print "\t\t%s %d %d %d\tprecision_score %s\trecall_score %s" % (
            label_dict.inverse_transform([i])[0], hit, test_cnt, pred_cnt, hit * 1.0 / (pred_cnt + 0.01),
            hit * 1.0 / (test_cnt + 0.01))


def get_feature_by_experienceList(workExperienceList, c_k_64_dic):
    level_two = [u'industry', u'department', u'type', u'position_name']
    feature_list = []
    for k in [0, -1]:
        for i in level_two:
            try:
                feature_list.append(c_k_64_dic[workExperienceList[k][i]])
            except Exception, e:
                feature_list.append(-1)
    return feature_list


level_one = [u'major', u'degree', u'gender', u'age', u'workExperienceList', u'_id', u'id']
level_two = [u'salary', u'end_date', u'industry', u'position_name', u'department', u'type', u'start_date', u'size']


def sentence_to_matrix_vec(sentence, model, featuresNum, k_mean_dict_1, k_mean_dict_2):
    temp = np.zeros((featuresNum * (7 * 5 + 3) + 7 * 5 * 2))
    if sentence == None: return temp

    num = (len(sentence) - 3) / 7 if (len(sentence) - 3) / 7 <= 5 else 5
    for i in range(num * 7):
        temp[featuresNum * i:featuresNum * (i + 1)] = model[sentence[i]]
        try:
            temp[38 * featuresNum + num * 2] = k_mean_dict_1[sentence[i]]
            temp[38 * featuresNum + num * 2 + 1] = k_mean_dict_2[sentence[i]]
        except Exception, e:
            continue
    for i in range(3):
        temp[(5 * 7 + i) * featuresNum:(5 * 7 + i + 1) * featuresNum] = model[sentence[-1 * (i + 1)]]
    return temp


def getAllFeatures(train, mapper):
    print "this is getAllFeatures"
    # every record has a cluster value calculated by lda
    w2c_f, w2c_w = 10, 14
    lda_dict_1 = util.read_dict(util.features_prefix + 'id_lda_256.pkl')
    lda_dict_2 = util.read_dict(util.features_prefix + 'id_lda_512.pkl')
    k_mean_dict_1 = util.read_dict(util.features_prefix + 'c_k_all_64.pkl')
    k_mean_dict_2 = util.read_dict(util.features_prefix + 'c_k_all_128.pkl')
    sentence_dict_path = util.txt_prefix + 'id_sentences.pkl'
    word2vec_path = util.txt_prefix + str(w2c_f) + 'features_1minwords_' + str(w2c_w) + 'context.pkl'
    sentence_dic = util.read_dict(sentence_dict_path)
    model = Word2Vec.load(word2vec_path)

    train_X = train[features]
    train_X = mapper.transform(train_X)  # .values
    new_train_X = []
    for i in xrange(len(train_X)):
        id = train_X[i][0]
        lda_1 = lda_dict_1[id]
        lda_2 = lda_dict_2[id]
        s = sentence_dic.get(id)
        f = np.concatenate(([train_X[i][1:].astype(np.float32)],
                            [sentence_to_matrix_vec(s, model, w2c_f, k_mean_dict_1, k_mean_dict_2)]), axis=1)[0]
        f = np.concatenate(([f], [[lda_1, lda_2]]), axis=1)[0]
        new_train_X.append(f)
    new_train_X = np.array(new_train_X)
    return new_train_X


if __name__ == "__main__":
    train_Y = []
    train_X = []
    test_X = []
    import os

    train_X = getAllFeatures(train, mapper)
    if os.path.exists(util.features_prefix + "/position_XY.pkl") is False:
        train_Y = list(train["predict_position_name"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/position_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/position_XY.pkl")
        print len(train_X[0]), len(train_Y)
        print 95 + 380 + 7 * 5 * 2 + 2
        print train_X[0]

    if os.path.exists(util.features_prefix + "/degree_XY.pkl") is False:
        train_Y = list(train["predict_degree"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/degree_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/degree_XY.pkl")
        print len(train_X[0]), len(train_Y)

    if os.path.exists(util.features_prefix + "/size_XY.pkl") is False:
        train_Y = list(train["predict_size"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/size_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/size_XY.pkl")
        # 99 + 380 + 7*5*2 + 2
        print len(train_X[0]), len(train_Y)

    if os.path.exists(util.features_prefix + "/salary_XY.pkl") is False:
        train_Y = list(train["predict_salary"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/salary_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/salary_XY.pkl")
        99 + 380 + 7*5*2 + 2
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(np.array(train_X[:100]), np.array(train_Y[:100]))
        print clf.predict(np.array(train_X[100:200]))
        print train_Y[100:200]
        from sklearn.feature_selection import SelectFromModel

        model = SelectFromModel(clf, prefit=True)
        list_1 = model.get_support()
        for i in range(len(list_1)):
            if list_1[i] == True:
                print i
    print 'pickle end'
