#! /usr/bin/env python
# -*- coding=utf-8 -*-
# @Author Leo
import util
import numpy as np

a = np.matrix([[1, 2], [2, 4]])
b = a.reshape([1, 4])
c = np.array([1, 2, 3, 4])
print np.array(b)
print c.shape
