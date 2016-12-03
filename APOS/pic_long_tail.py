#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import util


# list_ = [1, 3, 6, 7]
# for rank, column in enumerate(list_):
#     print rank, column

def get_sum_list(word_list):
    k_v = Counter(word_list)
    v = k_v.values()
    v.sort()
    v.reverse()
    sum_list = [sum(v[:i + 1]) for i in range(len(v))]
    return sum_list


def pic(count, now_list):
    sums = now_list[:count]
    sums = [0] + sums
    total = max(now_list)
    x = np.array(range(count + 1))
    y = np.array([0.0] + [float(sums[index]) / total for index in range(count)])
    return x, y


json_list = util.read_json('Job/data/resume_clean.json')
position_list = []
for i in xrange(len(json_list)):
    jobs = json_list[i]["workExperienceList"]
    for job_id in xrange(len(jobs)):
        job = jobs[job_id]
        if job is not None:
            position_name = job["position_name"]
            position_list.append(position_name)
file_path = ['Pos/data/oct27.conll']
file_lines = open(file_path[0]).readlines()
word_list = [line.split('	')[0] for line in file_lines]

file_path = ['Pos/data/daily547.conll']
file_lines = open(file_path[0]).readlines()
word_list2 = [line.split('	')[0] for line in file_lines]

max_num = 1000 / 2
start_num = 0
x, y = pic(max_num, get_sum_list(position_list))
xx, yy = pic(max_num, get_sum_list(word_list))
xxx, yyy = pic(max_num, get_sum_list(word_list2))
with plt.style.context('fivethirtyeight'):
    plt.plot(x[start_num:max_num], y[start_num:max_num], 'b')
    plt.plot(xx[start_num:max_num], yy[start_num:max_num], 'r')
    plt.plot(xxx[start_num:max_num], yyy[start_num:max_num], 'g')
    plt.text(max_num + 2, y[max_num] - 0.01, 'resume', fontsize=14, color="blue")
    plt.text(max_num + 2, yy[max_num] - 0.01, 'oct27', fontsize=14, color="red")
    plt.text(max_num + 2, yyy[max_num] - 0.01, 'daily547', fontsize=14, color="green")
plt.xlabel('The amount of most common tokens/positions')
plt.ylabel('The proportion of total')
plt.show()

print 'once', 'amount', 'proportion'
k_v = Counter(position_list)
y_one = len([value for value in k_v.values() if value == 1])
print y_one, len(k_v.values()), float(y_one) / len(k_v.values())

k_v = Counter(word_list)
y_one = len([value for value in k_v.values() if value == 1])
print y_one, len(k_v.values()), float(y_one) / len(k_v.values())

k_v = Counter(word_list2)
y_one = len([value for value in k_v.values() if value == 1])
print y_one, len(k_v.values()), float(y_one) / len(k_v.values())
