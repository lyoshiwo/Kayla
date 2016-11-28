#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import os
import pandas as pd
import numpy as np

file_list = os.listdir('temp')
print file_list
list_temp = []
for name in file_list:
    data_temp = pd.read_csv('temp/' + name)
    list_temp.append(data_temp['label'].values)
temp = np.zeros((len(list_temp[0])))
for i in list_temp:
    temp += i
temp /= len(list_temp)

wf = open("all.csv", "w")
wf.write("qid,uid,label\n")
rf = open('submit/final2.csv', 'r')
lines = rf.readlines()
count = 0
for line in lines[1:]:
    kk=line.split(',')
    wf.write(kk[0] + ','+kk[1]+',' + str(temp[count]) + '\n')
    count += 1

wf.close()
rf.close()
