# encoding=utf8
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys
from sklearn_pandas import DataFrameMapper
import pandas as pd
import util

reload(sys)
sys.setdefaultencoding('utf-8')
train = pd.read_csv('data/resume_clean.csv')
json_list = util.read_json(util.data_prefix + 'resume_clean.json')
features = list(train.keys())
print features, len(features)

print len(train), len(features)
train = pd.concat([train[features]])


def get_mapper(data_all):
    param_list = [
        # ('id', None),
        ('major', LabelEncoder()),
        ('age', None),
        ('gender', LabelEncoder()),
        ('degree', LabelEncoder()),
        # cut major into two parts if major can be cut
        ('major_1', LabelEncoder()),
        ('major_2', LabelEncoder()),

        # cut position into two parts if position can be cut
        ('last_position_name_1', LabelEncoder()),
        ('second_position_name_1', LabelEncoder()),
        ('first_position_name_1', LabelEncoder()),

        ('last_position_name_2', LabelEncoder()),
        ('second_position_name_2', LabelEncoder()),
        ('first_position_name_2', LabelEncoder()),
        # test by rule if there are English or words shows degrees
        ('isenglish', None),
        ('isjunior', None),
        ('isbachelor', None),
        ('ismaster', None),
        ('isintern', None),
        ('total_previous_job', None),
        ('last_size', None),
        ('last_salary', None),
        ('last_position_name', LabelEncoder()),
        ('last_start_year', None),
        ('last_start_month', None),
        ('last_end_year', None),
        ('last_end_month', None),
        ('last_interval_month', None),
        ('second_size', None),
        ('second_salary', None),
        ('second_position_name', LabelEncoder()),
        ('second_start_year', None),
        ('second_start_month', None),
        ('second_end_year', None),
        ('second_end_month', None),
        ('second_interval_month', None),
        ('first_size', None),
        ('first_salary', None),
        ('first_position_name', LabelEncoder()),
        ('first_start_year', None),
        ('first_start_month', None),
        ('first_end_year', None),
        ('first_end_month', None),
        ('first_interval_month', None),
        ('last_second_interval_month', None),
        ('diff_last_second_salary', LabelEncoder()),
        ('diff_last_second_size', LabelEncoder()),
        ('diff_last_second_position_name', LabelEncoder()),
        ('total_interval_month', None),
        ('diff_salary', LabelEncoder()),
        ('diff_size', LabelEncoder()),
        ('diff_position_name', LabelEncoder()),

        ('start_working_age', None),
        ('pre_working_month', None),
        ("pre_largest_size", None),
        ("pre_largest_salary", None),
        ("pre_least_size", None),
        ("pre_least_salary", None),
        ("pre_size1", None),
        ("pre_size2", None),
        ("pre_size3", None),
        ("pre_size4", None),
        ("pre_size5", None),
        ("pre_size6", None),
        ("pre_size7", None),
        ("pre_salary1", None),
        ("pre_salary2", None),
        ("pre_salary3", None),
        ("pre_salary4", None),
        ("pre_salary5", None),
        ("pre_salary6", None),
        ("pre_salary7", None),
    ]
    print "the mapper's param list is %s" % (len(param_list))
    mapper = DataFrameMapper(param_list)
    mapper.fit(data_all)
    return mapper


mapper = get_mapper(train)


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
    level_two = [u'position_name']
    feature_list = []
    for k in [0, -1]:
        for i in level_two:
            try:
                feature_list.append(c_k_64_dic[workExperienceList[k][i]])
            except Exception, e:
                feature_list.append(-1)
    return feature_list


level_one = [u'major', u'degree', u'gender', u'age', u'workExperienceList', u'_id', u'id']
level_two = [u'salary', u'end_date', u'position_name', u'start_date', u'size']


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


if __name__ == "__main__":
    from sklearn import preprocessing
    train_X = train[features]
    train_X = mapper.transform(train_X)
    print train_X.shape
    all = 0
    for i in range(train_X.shape[1]):
        all += train_X[:, i].max()
    print all
    print train_X.min()
    for i in xrange(train_X.shape[0]):
        for z in xrange(train_X.shape[1]):
            if train_X[i][z] < 0:
                print i, z, train_X[i][z], mapper.features[z]
                train_X[i][z] = 0
            train_X[i][z] = int(train_X[i][z])


    def get_y_by_name(name):
        y = train[name]
        return LabelEncoder().fit(list(y.values)).transform(list(y.values))


    y_position = get_y_by_name('predict_position_name')
    # y_size = get_y_by_name('predict_size')
    # y_salary = get_y_by_name('predict_salary')
    # pd.to_pickle([train_X, y_position, y_size, y_salary], 'pickle/manual_position_size_salary.pkl')
    enc = preprocessing.OneHotEncoder()
    enc.fit(train_X)
    X_data = enc.transform(train_X)
    print X_data[0]
    X_data = train_X
    param = dict()
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['eval_metric'] = 'merror'
    param['silent'] = 1
    param['min_child_weight'] = 10
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.2
    param['nthread'] = 2
    param['num_class'] = -1
    import xgboost as xgb
    import time

    train_Y = y_position
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    set_y = set(train_Y)
    param["num_class"] = len(set_y)
    dtrain = xgb.DMatrix(X_data, label=train_Y)
    param['objective'] = 'multi:softmax'
    xgb.cv(param, dtrain, 100, nfold=3, show_progress=True)
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
