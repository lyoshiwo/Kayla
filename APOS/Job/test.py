#! /usr/bin/env python
# -*- coding=utf-8 -*-
from step6_data_provider import DataProvider

if __name__ == "__main__":
    d = DataProvider(random_state=713, test_size=0.33)
    d.w2v_feature()
