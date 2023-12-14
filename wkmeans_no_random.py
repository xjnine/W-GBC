# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:06:34 2023

@author: xjnine
"""
import numpy as np
import random
import math

def InitCentroids(X, K):
    n = np.size(X, 0)
    first_idx = random.sample(range(n), 1)
    center = X[first_idx, :]
    dist_note = np.sqrt(np.sum((X - center) ** 2))
    next_idx = dist_note.argmax()
    next_idx = X[next_idx, :]
    centriod = []
    centriod.append(first_idx)
    centriod.append(next_idx)
    centriod = np.array(centriod)
    return centriod

def findClosestCentroids(X, w, centroids):
    K = np.size(centroids, 0)
    idx = np.zeros((np.size(X, 0)), dtype=int)
    n = X.shape[0]  # n 表示样本个数
    for i in range(n):
        subs = centroids - X[i, :]
        w_dimension2 = np.power(subs, 2)
        w_dimension2 = np.multiply(w, w_dimension2)
        w_distance2 = np.sum(w_dimension2, axis=1)
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(K)
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
    return idx


def computeCentroids(X, idx, K):
    n, m = X.shape
    centriod = np.zeros((K, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        s = np.sum(temp, axis=0)
        centriod[k, :] = s / np.size(index)
    return centriod


def computeWeight(X, centroid, idx, K, belta, w):
    n, m = X.shape
    weight = np.array(w)
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        distance2 = np.power((temp - centroid[k, :]), 2)
        D = D + np.sum(distance2, axis=0)
    e = 1 / float(belta)
    D = D + 0.0000001
    for j in range(m):
        temp = D[0][j] / D[0]
        temp_e = np.power(temp, e)
        temp_sum = np.sum(temp_e, axis=0)
        if temp_sum == 0:
            weight[0][j] = 0
        else:
            weight[0][j] = 1 / temp_sum
    return weight


def costFunction(X, K, centroids, idx, w, belta):
    n, m = X.shape
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        distance2 = np.power((temp - centroids[k, :]), 2)  # ? by m
        D = D + np.sum(distance2, axis=0)
    cost = np.sum(w ** belta * D)
    return cost


def isConvergence(costF, max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    for i in range(index - 1):
        if costF[i] < costF[i + 1]:
            return False
    if index >= max_iter:
        return True
    elif costF[index - 1] == costF[index - 2] == costF[index - 3]:
        return True
    elif abs(costF[index - 1] - costF[index - 2]) < 0.001:
        return True
    return 'continue'

def wkmeans(X, K, centroids, belta, max_iter, w):
    n, m = X.shape
    costF = []
    centroids = np.array(centroids)
    if w is None:
        r = np.ones((1, m))
        w = np.divide(r, r.sum())
    if n >= 9:
        belta = math.sqrt(n)
    else:
        belta = 3
    if max_iter != 1:
        for i in range(max_iter):
            idx = findClosestCentroids(X, w, centroids)
            w = computeWeight(X, centroids, idx, K, belta, w)
            c = costFunction(X, K, centroids, idx, 1, belta)
            costF.append(round(c, 4))
            if i < 2:
                continue
            flag = isConvergence(costF, max_iter)
            if flag == 'continue':
                continue
            elif flag:
                best_labels = idx
                best_centers = centroids
                isConverge = True
                return isConverge, best_labels, best_centers, costF, w
            else:
                isConverge = False
                return isConverge, None, None, costF, w
    else:
        idx = findClosestCentroids(X, w, centroids)
        w = computeWeight(X, centroids, idx, K, belta, w)
        best_labels = idx
        best_centers = centroids
        isConverge = True
        return isConverge, best_labels, best_centers, costF, w


class WKMeans:

    def __init__(self, n_clusters=3, max_iter=10, belta=7.0,centers=[], w=[]):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.belta = belta
        self.centers = centers
        self.w = w


    def fit(self, X):
        self.isConverge, self.best_labels, self.best_centers, self.cost, self.w = wkmeans(
            X=X, K=self.n_clusters, centroids=self.centers, max_iter=self.max_iter, belta=self.belta, w=self.w
        )
        return self

    def fit_predict(self, X, y=None):
        if self.fit(X).isConverge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'

    def get_params(self):
        return self.isConverge, self.n_clusters, self.belta, 'WKME'

    def get_cost(self):
        return self.cost