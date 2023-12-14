# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:24:35 2023

@author: hp
"""
from sklearn.preprocessing import MinMaxScaler

from W_GBC_split_by_k import *
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

def main():
    keys = ['Wine']
    data_path = '../data/'
    for d in range(len(keys)):
        K = 3
        df_x = pd.read_csv(data_path + keys[d] + "_X_1_0_" + ".csv", header=None)
        df_y = pd.read_csv(data_path + keys[d] + "_Y_1_0_" + ".csv", header=None, index_col=False)
        data = df_x.values
        y = df_y.values
        nn, _ = data.shape
        y = np.array(y).reshape((nn, 1))
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        data = np.hstack((data, y))
        indicators = []
        for i in range(20):
            new_list, data, indicator, true_label = hbc(data, y, K)
            indicators.append(indicator)
        indicators = np.array(indicators)
        n = indicators.shape[0]
        average = indicators.sum(axis=0) / n
        print(keys[d])
        print('[RI, NMI, ACC]')
        print(average)

if __name__ == '__main__':
    main()
