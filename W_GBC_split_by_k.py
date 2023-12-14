# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:11:52 2023

@author: hp
"""

import math
import warnings

from sklearn import metrics
import wkmeans_no_random

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import k_means
import numpy as np
import random

warnings.filterwarnings('ignore')


class GB:
    def __init__(self, data,
                 label):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.center = self.data.mean(
            0)  # According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        # self.init_center = self.random_center()  # Get a random point in each tag
        self.radius = self.get_radius()
        self.w_radius = 0
        self.flag = 0
        self.label = label
        self.num = len(self.data)
        self.out = 0
        self.size = 1
        self.overlap = 0  # 默认0的时候使用软重叠，1使用硬重叠
        self.hardlapcount = 0
        self.softlapcount = 0
        self.w = 1
        # self.noise = 0

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)

    def get_radius_by_w(self, w):
        return max(((self.data - self.center) ** 2 * w).sum(axis=1) ** 0.5)

# 计算子球的质量,hb为粒球中所有的点
def get_dm(hb, w):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    w_mat = np.tile(w, (num, 1))
    # 将w形状化为与diff_mat一致，然后做内积
    sq_diff_mat = diff_mat ** 2 * w_mat
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = 0
    sum_radius = sum(distances)
    if num > 1:
        return sum_radius / num
    else:
        return 1

def division(hb_list, hb_list_not, division_num, K):
    gb_list_new = []
    i = 0
    split_threshold = K
    for hb in hb_list:
        hb_no_w = hb
        if division_num != 1:
            parent_w = np.array([hb[-1][:]])
            hb_no_w = np.delete(hb, -1, axis=0)
            K = 2
            split_threshold = 2
        else:
            n, m = hb.shape
            w = np.ones((1, m))
            parent_w = w
        if len(hb_no_w) > split_threshold:
            i = i + 1
            if division_num != 1:
                K = 2
            ball, child_w = spilt_ball_by_k(hb_no_w, np.delete(parent_w,-1,axis=1), K, division_num)
            flag = 0
            for i in range(len(ball)):
                if len(ball[i]) == 0:
                    flag = 1
                    break
            # 分裂成功
            if flag == 0:
                # 子球的dm
                dm_child_ball = []
                # 子球的大小
                child_ball_length = []
                dm_child_divide_len = []
                for i in range(K):
                    temp_dm = get_dm(np.delete(ball[i],-1,axis = 1), child_w)
                    temp_len = len(ball[i])
                    dm_child_ball.append(temp_dm)
                    child_ball_length.append(temp_len)
                    dm_child_divide_len.append(temp_dm * temp_len)
                w0 = np.array(child_ball_length).sum()
                dm_child = np.array(dm_child_divide_len).sum() / w0
                dm_parent = get_dm(np.delete(hb_no_w,-1,axis = 1), np.delete(parent_w,-1,axis = 1))
                t2 = (dm_child < dm_parent)
                if t2:
                    child_w = child_w.flatten()
                    child_w_all = np.append(child_w,0)
                    for i in range(K):
                        temp_ball = np.append(ball[i], [child_w_all], axis=0)
                        gb_list_new.extend([temp_ball])
                else:
                    hb_list_not.append(hb)
            # 分裂失败
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)

    return gb_list_new, hb_list_not

def spilt_ball_by_k(data_no_w, w, k, division_num):
    centers = []
    max_iter = 1
    data = np.delete(data_no_w, -1, axis=1)
    if division_num != 1 or k == 2:
        k = 2
        center = data.mean(0)
        p_max1 = np.argmax(((data - center) ** 2).sum(axis=1) ** 0.5)
        p_max2 = np.argmax(((data - data[p_max1]) ** 2).sum(axis=1) ** 0.5)
        c1 = (data[p_max1] + center) / 2
        c2 = (data[p_max2] + center) / 2
        # 有初始质心
        centers.append(c1)
        centers.append(c2)
    else:
        centers = k_means(data, k, init="k-means++", n_init=10, random_state=42)[0]
        max_iter = 10
    model = wkmeans_no_random.WKMeans(n_clusters=k, max_iter=max_iter, belta=10, centers=centers, w=w)
    cluster = model.fit_predict(data)
    w = model.w
    ball = []
    for i in range(k):
        ball.append(data_no_w[cluster == i, :])
    return ball, w

def get_radius_by_w(hb, w):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    w_mat = np.tile(w, (num, 1))
    sq_diff_mat = diff_mat ** 2 * w_mat
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius

def InitCentroids(ball_centers, ball_weights, ball_radius, K, is_random=True):
    n, m = ball_centers.shape
    # 1. 随机获取聚类中心
    first_idx = random.sample(range(n), 1)
    center = ball_centers[first_idx]
    center_weights = ball_weights[first_idx]
    # 2. 计算每个样本距每个中心点的距离
    dist_note = np.zeros(n)
    dist_note += 1000000000.0

    for j in range(K):
        if j + 1 == K:
            break  # 已经计算了足够的聚类中心，直接退出
        # 计算每个样本和各聚类中心的距离，保存最小的距离
        for i in range(n):
            weight = (center_weights[j] + ball_weights[i]) / 2
            # 需要分为聚类点在球内还是求外
            temp_dist = np.sqrt(np.sum((center[j] - ball_centers[i]) ** 2 * weight))
            # temp_dist = np.sqrt(np.sum((center[j] - ball_centers[i]) ** 2))
            if temp_dist > ball_radius[i]:
                dist = temp_dist - ball_radius[i]
            else:
                dist = temp_dist
            # dist = np.sqrt(np.sum((center[j] - ball_centers[i]) ** 2 * weight)) - ball_radius[i]
            # 如果当前样本有更近的中心了，就更新距离
            if dist < dist_note[i]:
                dist_note[i] = dist
        # 若使用轮盘赌，根据距离远近随机产生新的聚类中心,否则使用最远距离的样本作为下一个聚类中心点
        if is_random:
            dist_p = dist_note / dist_note.sum()
            # 从range(n)中取概率为dist_p的一个点
            # 被分裂为一个球的处理
            if len(dist_p) == 1:
                next_idx = dist_note.argmax()
                center = np.vstack([center, ball_centers[next_idx]])
                center_weights = np.vstack([center_weights, ball_weights[next_idx]])
            else:
                # print(dist_p)
                next_idx = np.random.choice(range(n), 1, p=dist_p)
                center = np.vstack([center, ball_centers[next_idx]])
                center_weights = np.vstack([center_weights, ball_weights[next_idx]])
        else:
            # 修改初始中心点的选择
            next_idx = dist_note.argmax()
            center = np.vstack([center, ball_centers[next_idx]])
            center_weights = np.vstack([center_weights, ball_weights[next_idx]])
    return center, center_weights


def findClosestCentroids(ball_centers, ball_weights, ball_radius, centroids, centroids_weights):
    K = np.size(centroids, 0)
    idx = np.zeros((np.size(ball_centers, 0)), dtype=int)
    n = ball_centers.shape[0]  # n 表示样本个数
    for i in range(n):
        subs = centroids - ball_centers[i, :]
        dimension2 = np.power(subs, 2)
        ball_w = ball_weights[i, :]
        num = subs.shape[0]
        ball_w = np.tile(ball_w, (num, 1))
        w = (ball_w + centroids_weights) / 2
        w_dimension2 = np.multiply(w, dimension2)
        dis = np.sqrt(np.sum(w_dimension2, axis=1))
        w_distance2 = dis - ball_radius[i]
        index = w_distance2 < 0
        w_distance2[index] = dis[index]
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(K)
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
    return idx

def computeCentroids(ball_centers, ball_weights, idx, K):
    n, m = ball_centers.shape
    centriod = np.zeros((K, m), dtype=float)
    centriod_weights = np.zeros((K, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]  # 一个簇一个簇的分开来计算
        temp = ball_centers[index, :]
        temp_weight = ball_weights[index, :]
        # axis = 0 每一列（维）相加
        s = np.sum(temp, axis=0)
        s_weight = np.sum(temp_weight, axis=0)
        centriod[k, :] = s / np.size(index)
        centriod_weights[k, :] = s_weight / np.size(index)
    return centriod, centriod_weights


def costFunction(ball_centers, ball_weights, K, centroids, idx):
    n, m = ball_centers.shape
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = ball_centers[index, :]
        temp_w = ball_weights[index, :]
        # print(temp_w)
        distance2 = np.power((temp - centroids[k, :]), 2) * temp_w
        D = D + np.sum(distance2, axis=0)
    cost = np.sum(D)
    return cost


def isConvergence(costF, max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    # for i in range(index - 1):
    #     if costF[i] < costF[i + 1]:
    #         return False
    if index >= max_iter:
        return True
    elif costF[index - 1] == costF[index - 2]:
        return True
    return 'continue'

def get_radius_hb(data,center):
    data = np.delete(data,-1,axis=1)
    return max(((data - center) ** 2).sum(axis=1) ** 0.5)

def connect_ball_by_k_means(hb_list, K, max_iter, ave_w):
    hb_cluster = {}
    ball_data_list = []
    for i in range(0, len(hb_list)):
        hb = GB(hb_list[i], i)
        hb.w = ave_w
        hb.center = np.multiply(np.delete([hb.center],-1,axis=1), ave_w)
        hb.w_radius = get_radius_hb(hb_list[i], hb.center)
        hb_cluster[i] = hb
        temp = hb.center
        temp = np.append(temp, hb.w)
        temp = np.append(temp, hb.w_radius)
        ball_data_list.append(temp)
    ball_data_array = np.array(ball_data_list)
    n, m = ball_data_array.shape
    l = int((m - 1) / 2)
    # 获取球心的数据
    ball_centers = ball_data_array[:, 0:l]
    # 获取对应的权重
    ball_weights = ball_data_array[:, l:m - 1]
    # 获取对应的半径
    ball_radius = ball_data_array[:, -1]
    centroids, centroids_weights = InitCentroids(ball_centers, ball_weights, ball_radius, K)
    costF = []
    for i in range(max_iter):
        idx = findClosestCentroids(ball_centers, ball_weights, ball_radius, centroids, centroids_weights)
        centroids, centroids_weights = computeCentroids(ball_centers, ball_weights, idx, K)
        c = costFunction(ball_centers, ball_weights, K, centroids, idx)
        costF.append(round(c, 4))
        if i < 2:
            continue
        flag = isConvergence(costF, max_iter)
        if flag == 'continue':
            continue
        else:
            break
    for i in range(0, len(hb_cluster)):
        hb_cluster[i].label = idx[i]
    return hb_cluster, centroids


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    # assert用于判断一个表达式，在表达式结果为 False 的时候触发异常。若表达式结果为True，则不做任何反应
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    # print(D)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    # 利用匈牙利算法进行标签的再分配
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def hbc(data_all, y, K):
    hb_list_temp = [data_all]
    hb_list_not_temp = []
    division_num = 0
    # 按照dm值分裂
    while 1:
        ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
        division_num = division_num + 1
        hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp, division_num, K)
        ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
        if ball_number_new == ball_number_old:
            hb_list_temp = hb_list_not_temp
            break
    dic_w = {}
    ball_list = []
    w_total = np.zeros((1, data_all.shape[1] - 1 ))
    for index, hb_all in enumerate(hb_list_temp):
        w = np.delete([hb_all[-1,:]],-1, axis = 1)
        hb_no_w = np.delete(hb_all, -1, axis=0)
        ball_list.append(hb_no_w)
        dic_w[index] = w
        w_total = w_total + w
    ave_w = w_total / len(hb_list_temp)
    true_label = []
    max_data = 0
    max_indicator = []
    for i in range(20):
        max_iter = 30
        hb_cluster, centroids = connect_ball_by_k_means(ball_list, K, max_iter, ave_w)
        labels_true = []
        labels_pred = []
        for i in range(0, len(hb_cluster)):
            for j in range(len(hb_cluster[i].data)):
                labels_true.append(int(hb_cluster[i].data[j][-1]))
                labels_pred.append(int(hb_cluster[i].label))
        labels_pred = np.array(labels_pred)
        labels_true = np.array(labels_true)
        RI = metrics.cluster.adjusted_rand_score(labels_true, labels_pred)
        NMI = normalized_mutual_info_score(labels_true, labels_pred)
        ACC = acc(labels_true, labels_pred)
        indicator = [RI, NMI, ACC]
        indicator_sum = np.array(indicator).sum()
        if max_data < indicator_sum:
            true_label = labels_true
            max_data = indicator_sum
            max_indicator = indicator
    hb_list = []
    for i in range(len(hb_cluster)):
        for j in range(len(hb_cluster.get(i).data)):
            temp = np.delete([hb_cluster.get(i).data[j]],-1,axis=1)
            hb_list.append(temp)
    return ball_list, data_all, max_indicator, true_label
