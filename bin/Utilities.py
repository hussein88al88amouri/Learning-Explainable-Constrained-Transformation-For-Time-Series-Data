"""
Utilities.py

List of python function necessary for visualization, clusterin, etc.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
from copkmeans.cop_kmeans import cop_kmeans
import os
from tslearn.clustering import TimeSeriesKMeans
from  umap import umap_ as umap
import seaborn as sns
from sklearn.decomposition import PCA
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import matplotlib.patches as patches
from tslearn.metrics import cdist_dtw
from tslearn.metrics import dtw
import dtw as dtw_python
from tqdm import tqdm
import CopKmeans_DBA_Test as CopKmeansDBA
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import pandas as pd


def labelsToint(labels):
    tempdict = dict(enumerate(np.unique(labels)))
    labb = np.empty(labels.shape[0])
    for i in tempdict.items():
        labb[ labels == i[1] ] = i[0]
    return labb


def load_dataset_ts(ds_name, Type, path):
    ''' 
        ds_name: dataset name
        Type: either TRAIN or TEST 
        path: directory path to the dataset 
        Not that this only work with fixed time series length
    '''
    data, truelabels = load_from_tsfile_to_dataframe(
        os.path.join(path, f"{ds_name}/{ds_name}_{Type}.ts"))
    nsamples = data.shape[0]
    ndims = data.shape[1]
    slength = data['dim_0'][0].shape[0]
    temp = np.empty((nsamples, slength, ndims))
    for i in range(ndims):
             temp1 = data[f'dim_{i}'].values
             temp1 = np.concatenate([temp1[j].values
                                     for j in range(nsamples
                                    )]).reshape((nsamples, slength))
             temp[:, :, i] = temp1

    return temp, truelabels


def dtw_fast(s1, s2):
    ds = dtw(s1,s2)
    return ds


def DTWMax_dtwPython(data):
    sz = data.shape[0]
    maxdtw = 0
    for i in tqdm(range(sz - 1)):
        for j in range(i+1, sz):
            value = dtw_fast(data[i], data[j])
            if maxdtw < value:
                maxdtw = value
    return maxdtw


def DTWMax(data):
    sz = data.shape[0]
    blocksz = sz // 4 if sz % 100 == 0 else sz // 2
    ml = []
    pi = 0
    mlmax = 0
    for ci in range(blocksz, sz + blocksz, blocksz):
        dtw_ = cdist_dtw(data[pi:ci,:],data[pi:ci,:]) 
        dtw_[np.isinf(dtw_)] = 0
        mcmax = np.max(dtw_)
        if mcmax > mlmax:
            mlmax = mcmax
        pi = ci
    pi = 0
    for ci in range(blocksz, sz, blocksz):
        mcmax = DTWmaxpairwise(data[pi: ci, :],
                               data[pi + blocksz: ci+ blocksz, :])
        if mcmax > mlmax:
            mlmax = mcmax
    return mlmax


def DTWmaxpairwise(DL1,DL2):
    mlmax = 0
    for i in DL1:
        for j in DL2:
            cml = dtw_fast(i,j)
            if cml > mlmax:
                mlmax = cml
    return mlmax

def n_classes(y):
    return np.unique(y).shape[0]


def sortdata(data, truelabels):
    '''
        Return the sorted data according to the given labels
    '''
    sdata = []
    for i in np.unique(truelabels):
        for j in range(data.shape[0]):
            if truelabels[j] == i:
                sdata.append(data[j])
    return np.array(sdata), np.array(sorted(truelabels))


def plotTimeSeries(TS, sep=3, gap=20, ax=None,figsize=(10,5),color="tab10"):
    ts, dim = TS.shape
    ts1 = np.concatenate([TS, np.full((gap, dim) ,np.nan)]).reshape((-1), order='F')
    dim1 = np.array([[i] * (ts + gap) for i in range(dim)]).reshape((-1))
    TSdf = pd.DataFrame({'ts':ts1,'dim':dim1})
    scale = np.concatenate([np.arange(0, 1, 1 / (sep - 1)), np.array([1])])
    ymin = TSdf.ts.min() - 2
    ymax = TSdf.ts.max() + 2
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.lineplot(y=TSdf['ts'],x=TSdf.index, hue=TSdf['dim'],ax=ax,legend=False,linewidth=3, palette=[color])
    ax.set(ylabel = None)
    ax.set_xlim(0,ts)
    return ax


def getTSidx(data,TS):
    return np.argwhere(data.reshape((data.shape[0],
                                     -1),
                                    order='F') == TS.reshape((1, -1),
                                                             order='F'))[0,0]


def plotclusters(datasetname, data, truelabels, dir_=None, sample=False, samplenb=5,figsize=(10,5),color='tab10'):
    ''' 
    Plot the Timeseries in each clustering group
        dir: directory path to savee the clusters plot
    '''
    _, _, dim = data.shape
    nclusters = n_classes(truelabels)
    labels = np.unique(truelabels)
    colors = dict(zip(labels, sns.color_palette("tab10", nclusters).as_hex()))
    nm, _ = numSubplots(nclusters)
    plt.figure(constrained_layout=False,tight_layout=True)
    plt.suptitle(datasetname, fontsize=22)
    for j in range(nclusters):
        ax = plt.subplot(nm[0], nm[1], j+1)
        temp = data[truelabels == labels[j]]
        cs = temp.shape[0]
        if sample:
            idx = np.random.default_rng().choice(cs,size=samplenb,replace=False)
            for TS in temp[idx]:
                ax = plotTimeSeries(TS, sep=2, gap=30, ax=ax,figsize=(7, 5), color=colors[labels[j]])
            plt.title(f'Class {int(labels[j])}', y=0.9999, pad=3)
        else:
            for TS in temp:
                ax = plotTimeSeries(TS, sep=2, gap=30, ax=ax,figsize=(7, 5),color=color)
            plt.title('cluster:{}, csize:{}'.format(
                   j, cs), y=0.9999, pad=3, fontsize=14)
    fig = plt.gcf()
    patches = [ plt.plot([],[], ls="-", linewidth=4, color=sns.color_palette('tab10',dim)[i], 
            label=f"dim:{i}")[0] for i in range(dim) ]    
    fig.legend(handles=patches, bbox_to_anchor=(0.5, - 0.05),
               loc='lower center', ncol=dim, prop={'size': 12})
    fig.set_size_inches(15.5, 10.5, forward=True)
    fig.tight_layout()
    if dir_ != None:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        plt.savefig(f"{dir_}/{datasetname}_Clusters.pdf")
        plt.close('all')

def FDTWd(data):
    # '''Distance Map of DTW using dtiadistance package http://dtaidistance.readthedocs.io '''
    ns = data.shape[0]
    ds = cdist_dtw(data)
    ds[ds == np.inf] = 0
    return ds


def SdmaT(data_shtr):
    '''
    Distance Map approximation of DTW using Shapelet Transform 
        data_shtr: Shapelet Transformed dataset.
    '''
    sdm = np.zeros([data_shtr.shape[0], data_shtr.shape[0]])
    numb = data_shtr.shape[0] * (data_shtr.shape[0]+1)/2
    for ix in range(1, data_shtr.shape[0]):
        curr = ix * (ix + 1) / 2
        for jx in range(0, ix):
            sdm[ix, jx] = np.linalg.norm(data_shtr[ix, :] - data_shtr[jx, :])
            sdm[jx, ix] = sdm[ix, jx]
    return sdm


def DMAPplot(datasetname, dmap, labels, dir_=None):
    '''
    Distance Map plot with highliting the clusters in boxes
        dir_: directory to save the plots to.
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(dmap, cmap='hot', interpolation='None')
    ax.figure.colorbar(im, ax=ax)
    plt.title(datasetname, fontsize=30)
    lac = np.unique(labels, return_counts=True)
    indx = 1
    rectangles = {}
    for j in range(lac[0].shape[0]):
        rectangles[int(lac[0][j])] = patches.Rectangle((indx - 1, indx - 1),
                                                       lac[1][j] - 1, lac[1][j] - 1,
                                                       linewidth=3 if len(labels) < 20 else 5,
                                                       edgecolor='k', facecolor='none')
        indx += lac[1][j]
    for r in rectangles:
        ax.add_artist(rectangles[r])
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width()/2.0
        cy = ry + rectangles[r].get_height()/2.0
        ax.annotate(r, (cx, cy), color='magenta', weight='bold',
                    fontsize=15 if lac[0].shape[0] > 5 else 30, ha='center', va='center')
    fig.set_size_inches(10.5, 10.5, forward=True)
    fig.tight_layout()  
    if dir_ is not None:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        plt.savefig(f"{dir_}/{datasetname}_DTWmap", format='svg', dpi=1200)
        plt.close()


def constraint_generation_notarray(fr_, data, truelabels):
    ''' 
    Generation of constraints randomly (i,j, -1[CL]/+1[ML]) 
        fr_: fraction of datapoints to add constraints to
    '''
    import random
    xz = data.shape[0]
    nc = int(round(fr_*xz))
    C_ML = []
    C_CL = []
    ip = ()
    for idx in range(nc):
        i = random.randint(0, xz-1)
        j = random.randint(0, xz-1)
        while i == j:
            j = random.randint(0, xz-1)
        ip = tuple(sorted((i, j)))
        while ip in C_CL or ip in C_ML:
            i = random.randint(0, xz-1)
            j = random.randint(0, xz-1)
            while i == j:
                j = random.randint(0, xz-1)
            ip = tuple(sorted((i, j)))
        if ip not in C_CL and ip not in C_ML:
            if truelabels[ip[0]] == truelabels[ip[1]]:
                C_ML .append(ip)
            elif truelabels[ip[0]] != truelabels[ip[1]]:
                C_CL .append(ip)
    C_CL = np.array(C_CL)
    C_ML = np.array(C_ML)
    return C_ML, C_CL


def save_constraints(data, label, fr_, dirpath, dname,nb=5):
    ''' 
    Write constraints to file 
        dirpath: directory path to write the constraints
        dname: directory name
    '''
    dirc = f'{dirpath}/{dname}'
    if not os.path.exists(dirc):
        os.makedirs(dirc)
    for fr in fr_:
        for n in range(nb):
            C_ML = C_CL = []
            while len(C_CL) == 0 and len(C_ML) == 0:
                C_ML, C_CL = constraint_generation_notarray(fr, data, label)
            constraints = []
            np.save(os.path.join(dirc, '%s_fr_%.2f_ML_tn_%d.npy' %
                                    (dname, fr, n)), C_ML)
            np.save(os.path.join(dirc, '%s_fr_%.2f_CL_tn_%d.npy' %
                                    (dname, fr, n)), C_CL)


def KmeanClusteringTslearn(data, n_clusters, truelabels, metric='dtw',verbos=False):
    '''
    Kmeans clustering from TsLearn, can specify differet metrics for similarity calculation.
    '''
    nmis = []
    aris = []
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=100)
    pred_labels = km.fit_predict(data)
    nmis.append(normalized_mutual_info_score(truelabels, pred_labels))
    aris.append(adjusted_rand_score(truelabels, pred_labels))
    best_labels =pred_labels
    for i_trial in range(1, 10):
        km = TimeSeriesKMeans(n_clusters=n_clusters,
                              metric=metric, max_iter=100)
        pred_labels = km.fit_predict(data)
        nmis.append(normalized_mutual_info_score(truelabels, pred_labels))
        aris.append(adjusted_rand_score(truelabels, pred_labels))
        if verbos:
            print(f"KmeanClusteringTslearn {metric}"
                  f"[{i_trial}], nmi:{nmis[-1]}, ri:{aris[-1]}")
        if nmis[i_trial] > nmis[i_trial-1]:
            best_labels = pred_labels
    return best_labels, nmis, aris


def KMeanClustering(data, n_clusters, truelabels,verbos=False):
    '''Kmean clustering using sklearn '''
    nmis = []
    aris = []
    km = KMeans(n_clusters=n_clusters)
    pred_labels = km.fit_predict(data)
    nmis.append(normalized_mutual_info_score(truelabels, pred_labels))
    aris.append(adjusted_rand_score(truelabels, pred_labels))
    best_labels = pred_labels
    for i_trial in range(1, 10):
        km = KMeans(n_clusters=n_clusters, random_state=i_trial)
        pred_labels = km.fit_predict(data)
        nmis.append(normalized_mutual_info_score(truelabels, pred_labels))
        aris.append(adjusted_rand_score(truelabels, pred_labels))
        if verbos:
            print(f"KMeanClustering [{i_trial}],"
                  f"nmi:{nmis[-1]}, ri:{aris[-1]}")
        if nmis[i_trial] > nmis[i_trial-1]:
            best_labels = pred_labels
    return best_labels, nmis, aris


def COPKmeansClustering(data, truelabels, n_clusters, ml, cl,verbos=False):
    ''' 
    Constrained Kmean clustering using  https://github.com/Behrouz-Babaki/COP-Kmeans
        ml: Mustlink constraints 
        cl: Cannot link constraints
    '''
    nmis = []
    aris = []
    pred_labels = cop_kmeans(data, k=n_clusters, ml=ml, cl=cl)
    nmis.append(normalized_mutual_info_score(truelabels, pred_labels[0]))
    aris.append(adjusted_rand_score(truelabels, pred_labels[0]))
    print(f"COPKmeansClustering [{i_trial}],"
            f"nmi:{nmis[-1]}, ri:{aris[-1]}")
    best_labels = pred_labels[0]
    for i_trial in range(1, 10):
        pred_labels = cop_kmeans(data, k=n_clusters, ml=ml, cl=cl)
        nmis.append(normalized_mutual_info_score(truelabels, pred_labels[0]))
        aris.append(adjusted_rand_score(truelabels, pred_labels[0]))
        if verbos:
            print(f"COPKmeansClustering [{i_trial}],"
                  f"nmi:{nmis[-1]}, ri:{aris[-1]}")
        if nmis[i_trial] > nmis[i_trial-1]:
            best_labels = pred_labels [0]
    return best_labels, nmis, aris


def COPKmeansClusteringDBA(data, truelabels, n_clusters,
                           ml, cl, initialization='random',
                           max_iter=300, trial=10,
                           metric='dtw_distance', type_='dependent',
                           verbos=False):
    ''' 

    '''
    nmis = []
    aris = []
    pred_labels = CopKmeansDBA.cop_kmeans(data, k=n_clusters, ml=ml, cl=cl,
                                          metric=metric, type_=type_,
                                          max_iter=max_iter,
                                          initialization=initialization)
    nmis.append(normalized_mutual_info_score(truelabels, pred_labels[0]))
    aris.append(adjusted_rand_score(truelabels, pred_labels[0]))
            print(f"COPKmeansClustering [{i_trial}],"
                  f"nmi:{nmis[-1]}, ri:{aris[-1]}")
    best_labels = pred_labels[0]
    centers = pred_labels[1]
    for i_trial in range(1, trial):
        pred_labels = CopKmeansDBA.cop_kmeans(data, k=n_clusters, ml=ml, cl=cl,
                                              metric=metric, type_=type_,
                                              max_iter=max_iter,
                                              initialization=initialization)
        nmis.append(normalized_mutual_info_score(truelabels, pred_labels[0]))
        aris.append(adjusted_rand_score(truelabels, pred_labels[0]))
        if verbos:
            print(f"COPKmeansClustering [{i_trial}],"
                  f"nmi:{nmis[-1]}, ri:{aris[-1]}")
        if nmis[i_trial] > nmis[i_trial-1]:
            best_labels = pred_labels [0]
            centers = pred_labels [1]
    return best_labels, nmis, aris, centers


def ShapeletTransformData(data, model):
    '''
    Embedding of the TS in the new feature space using shapelet Transfrom.
    '''
    data_shtr = np.empty((data.shape[0], sum(model.shapelet_lengths.values())))
    for i in range(data.shape[0]):
        data_shtr[i] = model._shapelet_transform(data[i])
    return data_shtr


def FeatureSpaceUMAP(data, dataset, label, labelType,
                     alpha=None, gamma=None, fr=None,
                     algorithm=None, dirsave=None,
                     description=None, ml=None, cl=None,
                     random_state=None, palette="hot",
                     linestyle='.'):
    ''' 
    Visualization using UMAP in 2D space
    algorithm: string specify the algorithm used to predict the labels
    labelType: string specifying the typer of the labels (predicted / True)
    dirsave: directory to save the plot to. 
    Default (None) will not save the plots
    '''
    plt.rcParams['figure.constrained_layout.use'] = True
    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(data)
    textstr = '\n'.join((
        r'$\gamma=%.1f$' % (gamma, ),
        r'$\alpha=%.1f$' % (alpha, ),
        r'$fr=%.2f$' % (fr, ),)) if fr != None else ''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if fr:
        fig, axs = plt.subplots(2, 1)
        axs0, axs1 = axs 
        axs1.scatter(embedding[:, 0], embedding[:, 1],
                     c=[sns.color_palette(palette,
                                          np.unique(label).shape[0]+1)[x]
                        for x in list(label.astype(int))],
                     alpha=0.7)
        axs1.set_title("The constraint information is displayed. ")
        plt.suptitle(f"{dataset} {algorithm} {labelType} {description}")
        if ml.shape[0] != 0:
            for i in ml:
                axs1.plot([embedding[i[0], 0], embedding[i[1], 0]],
                          [embedding[i[0], 1], embedding[i[1], 1]],
                          color='black', linestyle='solid',
                          linewidth=5, alpha=0.8)
                axs1.scatter([embedding[i[0], 0], embedding[i[1], 0]],
                             [embedding[i[0], 1], embedding[i[1], 1]],
                             color='green', alpha=0.05)
        if cl.shape[0] != 0:
            for i in cl:
                axs1.plot([embedding[i[0], 0], embedding[i[1], 0]],
                          [embedding[i[0], 1], embedding[i[1], 1]],
                          color='black', linestyle='dotted',
                          linewidth=3, alpha=0.8)
                axs1.scatter([embedding[i[0], 0], embedding[i[1], 0]],
                             [embedding[i[0], 1], embedding[i[1], 1]],
                             color='red', alpha=0.05)
        axs1.text(min(embedding[:, 0]), max(embedding[:, 1]),
                  textstr, bbox=props, fontsize=10, verticalalignment='top')
    else:
        fig, axs0 = plt.subplots(1, 1)
    axs0.scatter(embedding[:, 0], embedding[:, 1],
                 c=[sns.color_palette(palette,
                                      np.unique(label).shape[0]+1)[x]
                    for x in list(label.astype(int))],
                 alpha=0.7)
    axs0.set_title("Data Points in the Feature Space.")
    axs0.text(min(embedding[:, 0]), max(embedding[:, 1]),
              textstr, bbox=props, fontsize=10, verticalalignment='top')
    if dirsave is not None:
        if not os.path.exists(dirsave):
            os.makedirs(dirsave)
        plt.savefig(f"{dirsave}/{dataset}_fr_{fr:.2f}"
                    f"alpha_{alpha:.1f}_gamma_{gamma:.1f}.png",
                    dpi=1200)
    return embedding


def FeatureSpacePCA(data, dataset, label, labelType,
                    alpha=None, gamma=None, fr=None,
                    n_components=2, algorithm=None,
                    dirsave=None, description=None,
                    ml=None, cl=None, palette="hot",
                    linestyle='.'):
    '''Visualization using PCA in either 2D space or 3D space
        algorithm: string specify the algorithm used to predict the labels
        labelType: string specifying the typer of the labels (predicted / True)
        save: bool to specify if to save the plot
        dirsave: directory to save the plot to.
        Default (None) will not save the plots.
    '''
    plt.rcParams['figure.constrained_layout.use'] = True
    pca = PCA(n_components=int(n_components)).fit(data)
    pca_d = pca.transform(data)
    print(pca_d.shape)
    textstr = '\n'.join((
        r'$\gamma=%.1f$' % (gamma, ),
        r'$\alpha=%.1f$' % (alpha, ),
        r'$fr=%.2f$' % (fr, ),)) if fr is not None else ''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if n_components == 3:
        print(n_components)
        fig = plt.figure()
        if fr:
            ax = fig.add_subplot(211, projection='3d')
            ax1 = fig.add_subplot(212, projection='3d')
            ax1.scatter(pca_d[:, 0], pca_d[:, 1], pca_d[:, 2],
                        c=[sns.color_palette(palette,
                                             np.unique(label).shape[0]+1)[x]
                           for x in list(label.astype(int))],
                        alpha=0.7)
            ax1.set_title("The constraint information is displayed. ")
            ax1.set_xlabel('PCA1')
            ax1.set_ylabel('PCA2')
            ax1.set_zlabel('PCA3')
            if ml.shape[0] != 0:
                for i in ml:
                    ax1.plot([pca_d[i[0], 0], pca_d[i[1], 0]],
                             [pca_d[i[0], 1], pca_d[i[1], 1]],
                             [pca_d[i[0], 2], pca_d[i[1], 2]],
                             color='black', linestyle='solid',
                             linewidth=5, alpha=0.8)
                    ax1.scatter([pca_d[i[0], 0], pca_d[i[1], 0]],
                                [pca_d[i[0], 1], pca_d[i[1], 1]],
                                [pca_d[i[0], 2], pca_d[i[1], 2]],
                                color='green', alpha=0.05)
            if cl.shape[0] != 0:
                for i in cl:
                    ax1.plot([pca_d[i[0], 0], pca_d[i[1], 0]],
                             [pca_d[i[0], 1], pca_d[i[1], 1]],
                             [pca_d[i[0], 2], pca_d[i[1], 2]],
                             color='black', linestyle='dotted',
                             linewidth=3, alpha=0.8)
                    ax1.scatter([pca_d[i[0], 0], pca_d[i[1], 0]],
                                [pca_d[i[0], 1], pca_d[i[1], 1]],
                                [pca_d[i[0], 2], pca_d[i[1], 2]],
                                color='red', alpha=0.05)
        else:
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_d[:, 0], pca_d[:, 1], pca_d[:, 2],
                   c=[sns.color_palette(palette,
                                        np.unique(label).shape[0]+1)[x]
                      for x in list(label.astype(int))],
                   alpha=0.7)
        ax.set_title("Data Points in the Feature Space.")
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        ax.text(min(pca_d[:, 0]), min(pca_d[:, 1]), max(
            pca_d[:, 2]), textstr, bbox=props, fontsize=10, zorder=1)
        plt.suptitle(f"{dataset} {algorithm} {labelType} {description}")
    else:
        fig = plt.figure()
        if fr:
            ax = fig.add_subplot(211)
            ax1 = fig.add_subplot(212)
            ax1.scatter(pca_d[:, 0], pca_d[:, 1],
                        c=[sns.color_palette(palette,
                                             np.unique(label).shape[0]+1)[x]
                           for x in list(label.astype(int))],
                        alpha=0.7)
            ax1.set_title("The constraint information is displayed.")
            ax1.set_xlabel('PCA1')
            ax1.set_ylabel('PCA2')
            if ml.shape[0] != 0:
                for i in ml:
                    ax1.plot([pca_d[i[0], 0], pca_d[i[1], 0]],
                             [pca_d[i[0], 1], pca_d[i[1], 1]],
                             color='black', linestyle='solid',
                             linewidth=5, alpha=0.8)
                    ax1.scatter([pca_d[i[0], 0], pca_d[i[1], 0]],
                                [pca_d[i[0], 1], pca_d[i[1], 1]],
                                color='green', alpha=0.05)
            if cl.shape[0] != 0:
                for i in cl:
                    ax1.plot([pca_d[i[0], 0], pca_d[i[1], 0]],
                             [pca_d[i[0], 1], pca_d[i[1], 1]],
                             color='black', linestyle='dotted',
                             linewidth=3, alpha=0.8)
                    ax1.scatter([pca_d[i[0], 0], pca_d[i[1], 0]],
                                [pca_d[i[0], 1], pca_d[i[1], 1]],
                                color='red', alpha=0.05)
        else:
            ax = fig.add_subplot(111)
        ax.scatter(pca_d[:, 0], pca_d[:, 1],
                   c=[sns.color_palette(palette,
                                        np.unique(label).shape[0]+1)[x]
                      for x in list(label.astype(int))],
                   alpha=0.7)
        ax.set_title("Data Points in the Feature Space.")
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.text(min(pca_d[:, 0]), max(pca_d[:, 1]), textstr,
                bbox=props, fontsize=10, verticalalignment='top')
        plt.suptitle(f"{dataset} {algorithm} {labelType} {description}")
    if dirsave is not None:
        if not os.path.exists(dirsave):
            os.makedirs(dirsave)
        plt.savefig(f"{dirsave}/{dataset}_fr_{fr:.2f}"
                    f"alpha_{alpha:.1f}_gamma_{gamma:.1f}.png",
                    dpi=1200)


def factor(n):
    """"""
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def numSubplots(n):
    """
    Calculate  subplots number of rows and columns
    Paramters
    -------
        n : int,  the desired number of subplots.
    Returns
    -------
        p : 2D int array [rows, columns]
        n : int, number of subplots.
        Example: neatly lay out 13 sub-plots
        >> p=numSubplots(13)
        >> p = np.array([3,5],dtype=int32)
        by Rob Campbell - January 2010
    """

    from sympy import isprime
    while isprime(n) & n > 4:
        n += 1
    p = factor(n)
    if len(p) == 1:
        p = [1, p[0]]
        return p, n
    while len(p) > 2:
        if len(p) >= 4:
            p[0] = p[0] * p[-2]
            p[1] = p[1] * p[-1]
            p[-2:] = []
        else:
            p[0] = p[0] * p[1]
            p[-2: -1] = []
        p.sort()
    while p[1] / p[0] > 2.5:
        N = n+1
        p, n = numSubplots(N)
    return p, n





































