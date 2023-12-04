# -*- coding: utf-8 -*-
#%%
import random
import numpy as np
from tslearn.barycenters import  dtw_barycenter_averaging as dtwDBA
from tslearn.metrics import dtw

#DTW  is adapted to multi dimentional, Euclidean to do
def cop_kmeans(dataset, k, ml=[], cl=[], metric='l2_distance', type_='dependent',
               initialization='kmpp',
               max_iter=100, tol=1e-4):

    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset,metric=metric)
    tol = tolerance(tol, dataset)
    centers = initialize_centers(dataset, k, initialization,metric=metric, type_=type_)
    centers = np.array(centers)
    iterr = 0
    for _ in range(max_iter):
        # print(f'Iter {iterr}')
        clusters_ = [-1] * len(dataset)
        clusters_ = np.array(clusters_)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, dataset[[i]],metric=metric, type_=type_)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1
                if not found_cluster:
                    return None, None
        clusters_, centers_  = compute_centers(clusters_, dataset,k, ml_info, metric=metric, type_=type_)
        # print(f'shift value {shift} tol value {tol}')
        # if shift <= tol:
        #     print(f'shift value {shift} tol value {tol}')
        #     break

        centers = centers_
        iterr = iterr + 1

    return clusters_, centers_

def dtw_distance(point1, point2,type_='dependent'):
    #timestamps are the rows and the multivariate features are the columns
    point1 = np.array(point1) if type(point1) != type(np.array([])) else point1
    point2 = np.array(point2) if type(point2) != type(np.array([])) else point2
    if type_ == 'dependent': 
        return dtw(point1[0],point2[0])
    elif type_ == 'independent':
        dtwdistance = 0
        for dim in range(point1.shape[2]):
            x = point1[0,:,dim]
            y = point2[0,:,dim]
            dtwdistance = dtwdistance +  dtw(x,y)          
        return dtwdistance
    
def l2_distance(point1, point2):
        from scipy.spatial.distance import cdist
        return cdist(point1[0, :, :].reshape((1,-1)), point2[0, :, :].reshape((1,-1)),'euclidean')[0][0]**2
        return np.linalg.norm(point1[0, :, :] - point2[0, :, :])**2
        summ = 0
        for ii in range(point1.shape[2]):
            summ += sum([(float(i)-float(j))**2 for (i, j) in zip(point1[0,:,ii], point2[0,:,ii])])
        return np.sqrt(summ)

# taken from scikit-learn (https://goo.gl/1RYPP5)
def tolerance(tol, dataset):
    n = len(dataset)
    tsLenght = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n))/float(n) for d in range(tsLenght)]
    variances = [sum((dataset[i][d]-averages[d])**2 for i in range(n))/float(n) for d in range(tsLenght)]
    return np.mean(tol * sum(variances) / tsLenght)

def closest_clusters(centers, datapoint, metric='l2_distance', type_='dependent'):
    if metric == 'l2_distance':
        distances = [l2_distance(centers[[center]], datapoint) for
                     center in range(len(centers))]
    elif metric == 'dtw_distance':
        distances = [dtw_distance(centers[[center]], datapoint, type_=type_) for
                     center in range(len(centers))]
        
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances

def initialize_centers(dataset, k, method,metric='l2_distance', type_='dependent'):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]

    elif method == 'kmpp':
        chances = [1] * len(dataset)
        centers = []
        kk = 0
        for _ in range(k):
            chances = [x/sum(chances) for x in chances]
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])
            if metric == 'l2_distance':
                centers_arr = np.array(centers)
                tsLength = dataset.shape[1]
                dim = dataset.shape[2]
                for index, point in enumerate(dataset):
                    cids, distances = closest_clusters(centers_arr, point.reshape(-1,tsLength,dim),metric=metric, type_=type_)
                    chances[index] = distances[cids[0]]

            elif metric == 'dtw_distance':
                centers_arr = np.array(centers)
                tsLength = dataset.shape[1]
                dim = dataset.shape[2]
                for index, point in enumerate(dataset):
                    cids, distances = closest_clusters(centers_arr, point.reshape(-1,tsLength,dim),metric=metric, type_=type_)
                    chances[index] = distances[cids[0]]
                
            kk = kk + 1
        return centers

def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False

def compute_centers(clusters, dataset, k, ml_info,metric='l2_distance', type_='dependent'):
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]

    tsLength = dataset.shape[1]
    dim = dataset.shape[2]

    counts = [0] * k_new
    if metric == 'l2_distance':
        clusters_arr = np.array(clusters)
        centers = np.zeros([k,tsLength,dim])
        for j, c in enumerate(clusters):
            for i in range(tsLength):
                for d in range(dim):
                    centers[c][i][d] += dataset[j][i][d]
            counts[c] += 1
    
        for j in range(k_new):
            for i in range(tsLength):
                for d in range(dim):
                    centers[j][i] = centers[j][i][d]/float(counts[j])

    elif metric == 'dtw_distance':
        clusters_arr = np.array(clusters)
        centers = np.zeros([k,tsLength,dim])
        for cluster_id in id_map.values():
            group = np.where(clusters_arr == cluster_id)[0]
            centers[cluster_id] = dtwDBA(dataset[group].reshape(-1, tsLength, dim))
            # counts[cluster_id] = group.shape[0]

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info

        if metric == 'l2_distance':
            current_scores = [sum(l2_distance(centers[[clusters[i]]], dataset[[i]])
                                  for i in group)
                              for group in ml_groups]
            group_ids = sorted(range(len(ml_groups)),
                               key=lambda x: current_scores[x] - ml_scores[x],
                               reverse=True)

        elif metric == 'dtw_distance':
            current_scores = [sum(dtw_distance(centers[[clusters[i]]], dataset[[i]], type_=type_)
                                  for i in group)
                              for group in ml_groups]
            group_ids = sorted(range(len(ml_groups)),
                               key=lambda x: current_scores[x] - ml_scores[x],
                               reverse=True)

        for j in range(k-k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    if metric =='dtw_distance':
        return np.array(clusters), np.array(centers)
    else:
        return clusters, centers
    
def get_ml_info(ml, dataset,metric='l2_distance', type_='dependent'):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]: continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False

    # averaging time series in groups to create the centriods
    # using euclidean distance, need to be adapted for multivariate
    if metric == 'l2_distance':

        nbcentroids, tsLength, dim = len(groups), dataset.shape[1], dataset.shape[2]
        centroids = np.zeros([nbcentroids, tsLength, dim])
        for j, group in enumerate(groups):
            for tsL in range(tsLength):
                for d in range(dim):
                    for i in group:
                        centroids[j][tsL][d] += dataset[i][tsL][d]
                    centroids[j][tsL][d] /= float(len(group))
        scores = [sum(l2_distance(centroids[[j]], dataset[[i]])
                      for i in groups[j])
                  for j in range(len(groups))]
    
    elif  metric == 'dtw_distance':
        nbcentroids, tsLength, dim = len(groups), dataset.shape[1], dataset.shape[2]
        centroids = np.zeros([nbcentroids, tsLength, dim])
        for j, group in enumerate(groups):
            centroids[j] = dtwDBA(dataset[group].reshape(-1, tsLength, dim))
        scores = [sum(dtw_distance(centroids[[j]], dataset[[i]], type_=type_)
                      for i in groups[j])
                  for j in range(len(groups))]

    return groups, scores, centroids

def transitive_closure(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' %(i, j))

    return ml_graph, cl_graph

