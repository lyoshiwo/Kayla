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
