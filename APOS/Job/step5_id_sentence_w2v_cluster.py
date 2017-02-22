# encoding=utf8
import jieba
import re
import os
from util import read_dict, write_dic, read_json
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import util
import datetime


def work_time(start_date, end_date):
    start_date = start_date.strip()
    end_date = end_date.strip()
    if u'今' in end_date or 'Present' in end_date or u'其他' in end_date:
        end_date = '2015-05'
    a = str(start_date).split('-')[0]
    b = str(start_date).split('-')[1]
    c = str(end_date).split('-')[0]
    d = str(end_date).split('-')[1]
    d1 = datetime.datetime(int(c), int(d), 1)
    d2 = datetime.datetime(int(a), int(b), 1)
    m = int(round(float((d1 - d2).days) / 30.5, 2) / 3)
    # 工作了多少个季度
    return 'm' + str(m)


def get_ids_sentence():
    file_path_train = util.data_prefix + r'resume_clean.json'
    json_all = read_json(file_path_train)
    level_zero = [u'size', u'salary']
    level_two = [u'position_name']
    level_one = [u'major', u'gender', u'age']
    dic_all = {}
    count2 = 0
    all = 0
    for i in json_all:
        id_one = i[u'id']
        workExperienceList = i[u'workExperienceList']
        all += 1
        sentence = []
        count = 0
        for w in workExperienceList:
            if w is None:
                continue
            if len(w) > 2:
                count2 += 1
            count += 1
            if count is 1:
                continue
            for s in level_zero:
                try:
                    if w[s] is None:
                        sentence.append(s + u'1')
                    else:
                        sentence.append(s + str(w[s]))
                except Exception, e:
                    sentence.append(s + u'1')
            for t in level_two:
                try:
                    if w[t] is None:
                        sentence.append(u'其他')
                    else:
                        sentence.append(w[t])
                except Exception, e:
                    sentence.append(u'其他')
            try:
                sentence.append(work_time(w['start_date'], w['end_date']))
            except Exception, e:
                sentence.append(u'm4')
        for z in level_one:
            if z is u'age':
                try:
                    sentence.append(u'a' + str(i[u'age']))
                except Exception, e:
                    sentence.append(u'a24')
                continue
            try:
                if i[z] is None:
                    sentence.append(u'其他')
                else:
                    sentence.append(i[z])
            except Exception, e:
                print e.message
        dic_all.setdefault(id_one, sentence)
    print len(dic_all)
    # 把 id对应的sentences存起来
    util.write_dic(dic_all, 'pickle/' + 'id_sentences.pkl')
    return dic_all


if __name__ is "__main__":
    sentence_dict_path = 'pickle/' + 'id_sentences.pkl'
    if os.path.exists(sentence_dict_path) is False:
        get_ids_sentence()
    exit()
    if os.path.exists('pickle/' + 'id_texts.pkl') is False:
        id_sentence = read_dict(sentence_dict_path)
        print len(id_sentence)
        id_text = {}
        for i in id_sentence.keys():
            sentence = id_sentence[i]
            temp = ' '.join(sentence)
            temp = re.sub('-|\\)|\\(|（|/|）', ' ', temp).replace('）', '')
            cut_str = jieba.cut(temp)
            text = " ".join(cut_str)
            text = re.sub(r'\s{2,}', ' ', text)
            id_text.setdefault(i, (text.replace('（', '')).split(' '))
        write_dic(id_text, 'pickle/' + 'id_texts.pkl')
    id_text = read_dict('pickle/' + 'id_texts.pkl')
    texts = id_text.values()
    features, words = 10, 14
    if os.path.exists('pickle/' + str(features) + 'features_1minwords_' + str(words) + 'text.pkl') is False:
        # Set values for various parameters
        num_features = features  # Word vector dimensionality
        min_word_count = 1  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = words  # Context window size
        down_sampling = 1e-3  # Down_sample setting for frequent words

        print "Training Word2Vec model..."
        model = Word2Vec(texts, workers=num_workers, \
                         size=num_features, min_count=min_word_count, \
                         window=context, sample=down_sampling, seed=1, negative=3)
        model.init_sims(replace=True)
        word2vec_path = 'pickle/' + str(features) + 'features_1minwords_' + str(words) + 'text.pkl'
        model.save(word2vec_path)

    if os.path.exists('pickle/c_v_all.pkl') is False:
        print 'create c_v_all'
        import numpy

        word2vec_path = 'pickle/' + str(features) + 'features_1minwords_' + str(words) + 'text.pkl'
        model = Word2Vec.load(word2vec_path)
        id_sentence = read_dict(sentence_dict_path)
        sentence = id_sentence.values()
        c_vec = {}
        for s in sentence:
            for section in s:
                used = 0
                temp_vec = numpy.zeros(features)
                if c_vec.has_key(section):
                    continue
                else:
                    try:
                        temp_vec += model[section]
                        c_vec.setdefault(section, temp_vec)
                        print section, temp_vec[0:2]
                    except Exception, e:
                        for ww in jieba.cut(section):
                            try:
                                temp_vec += model[ww]
                                used += 1
                            except Exception, e:
                                continue
                        if used is 0:
                            used = 1
                        c_vec.setdefault(section, temp_vec / used)
                        print section, (temp_vec / used)[0:2]
        write_dic(c_vec, 'pickle/' + 'c_v_all.pkl')
    k_clusters = 128
    if os.path.exists('pickle/' + 'cluster_two_' + str(k_clusters) + '.pkl') is False:
        print 'create c_k_all'
        c_vec = read_dict('pickle/' + 'c_v_all.pkl')
        c_key = c_vec.keys()
        vec_set = c_vec.values()
        KMeans_model = KMeans(n_clusters=k_clusters, n_init=5)
        KMeans_model.fit(vec_set)
        k_labels = KMeans_model.labels_
        dict_temp = {}
        for index in range(len(c_vec)):
            dict_temp.setdefault(str(c_key[index]).decode('utf-8'), k_labels[index])
            if 13 > k_labels[index] > 10:
                print c_key[index], k_labels[index]
        print len(dict_temp)
        write_dic(dict_temp, 'pickle/' + 'cluster_two_' + str(k_clusters) + '.pkl')
    k_clusters = 64
    if os.path.exists('pickle/' + 'cluster_two_' + str(k_clusters) + '.pkl') is False:
        print 'create c_k_all'
        c_vec = read_dict('pickle/' + 'c_v_all.pkl')
        c_key = c_vec.keys()
        vec_set = c_vec.values()
        KMeans_model = KMeans(n_clusters=k_clusters, n_init=5)
        KMeans_model.fit(vec_set)
        k_labels = KMeans_model.labels_
        dict_temp = {}
        for index in range(len(c_vec)):
            dict_temp.setdefault(c_key[index], k_labels[index])
            if 13 > k_labels[index] > 10:
                print c_key[index], k_labels[index]
        print len(dict_temp)
        write_dic(dict_temp, 'pickle/' + 'cluster_two_' + str(k_clusters) + '.pkl')
        c_k_all = read_dict('pickle/' + 'cluster_two_' + str(k_clusters) + '.pkl')
        set_cluster = set(c_k_all.values())
        set_cluster = list(set_cluster)
        flg = range(10)
        for i in flg:
            for k in c_k_all.keys():
                if set_cluster[i] is c_k_all[k]:
                    print set_cluster[i], k
