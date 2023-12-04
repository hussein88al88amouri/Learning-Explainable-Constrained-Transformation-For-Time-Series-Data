# %%  Adding bin to search path
import sys
sys.path.append("/home/elamouri/CDPS/bin")
#from CDPS.bin.Utilities import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Utilities as utls
import NN_CLDPS_Torch_Multi as CLDPSM
from glob import glob
import Quality_measures as qm
import QMplots as qmplots
import os
#%% Objective
'''
1- Use one good dataset with CDPS (CBF, trace) and one where it is not that good (TwoPatterns,  Meat)
2- Obtain the embedings
3- Obtain clusters for  CDPS using Kmeans. And clusters of Raw dataset using DBA Copkmeans
4- Order the shapelets   with respect to CDPS, and then another order with respect to raw
5- plot UMAP/TSNE plots for the embedings, 
    + Ideas for the basis:
        a- using the UMAP/TSNE dimension reduction on all shapelets
            i- on the first 10 informative shapelets (single, successive, combined ) under (local, global) respectively
        b- using the first two shapelets extracted from (single, successive, combined ) under (local, global) respectively
    + color the constrainted pair of points and put a link between them
    + put a circle around the points that belong to the same cluster/ color the cluster
    
''
'''

#%% loading Datasets
# remove CharacterTrajectories
Order = 'Univariate'
dpath = f'/home/elamouri/{Order}_ts/' 
#%%
# from glob import glob
# datasets = glob(f'{dpath}/*')
# d=''
# for dataset in datasets:
#     dataset = os.path.basename(dataset)
#     if dataset != 'CharacterTrajectories' and dataset != 'InsectWingbeat' and dataset != 'JapaneseVowels' and dataset != 'SpokenArabicDigits' :        
#         d = d+f' {dataset}'
# print(f'{dataset}')
#%%
dataset = 'SyntheticControl'
Trdata, Trgt_labels = utls.load_dataset_ts(dataset, 'TRAIN', dpath)
Tsdata, Tsgt_labels = utls.load_dataset_ts(dataset, 'TEST', dpath)
alldata = True
if alldata:
    data = np.concatenate((Trdata, Tsdata))
    labels = np.concatenate((Trgt_labels, Tsgt_labels))
else:
    data = Trdata
    del Trdata
    labels = Trgt_labels
    del Trgt_labels
labels = utls.labelsToint(labels)
n_clusters = utls.n_classes(labels)
print(f' dataset shpae : {data.shape}')
#%% Loading Embedings, model (Shapelets; Constraints)
embs = {}
#%%
fr=0.25
rootPath = f'/home/elamouri/ECMLPaperResults/TestsForPaper/TestsCDPSUNIAlldata/MUL/{dataset}/alpha_2.5_gamma_2.5_fr_{fr}_nfr_1/lmin_0.15_shapelet_max_scale_3_ratio_n_shapelets_10'
#rootPath = f'/home/elamouri/ECMLPaperResults/TestsForPaper/TestsCDPSUNIAlldata/MUL/{dataset}/Noconstraints/lmin_0.15_shapelet_max_scale_3_ratio_n_shapelets_10'
modelPath = f'{rootPath}/Model/Final*'
model = CLDPSM.CLDPSModel.model_load(glob(modelPath)[0],usepickle=True) #use pickle if u have a magic error, since u saved with pickle
ml, cl = model.ML, model.CL
alpha, gamma, fr = model.loss_.alpha, model.loss_.gamma, model.loss_.fr
embeddingsPath = f'{rootPath}/Embedings/*'
embs[f'{fr}'] = np.load(glob(embeddingsPath)[0])
#%%
embeddings = np.load(glob(embeddingsPath)[0])
n, t, d = embeddings.shape[0], embeddings.shape[1], 1 if len(embeddings.shape) == 2 else embeddings.shape[2]
#%% clustering datapoints
clusteringLabels =  {}
clusteringNMI =  {}
clusteringARI = {}
centers = {}
ExpOnEmb = True
type_ = 'dependent'
if ExpOnEmb:
    clusteringLabels['Emb'], clusteringNMI['Emb'], clusteringARI['Emb'], centers['Emb'] = utls.COPKmeansClusteringDBA(embeddings.reshape(n,t,d), labels, n_clusters, [], [], 
                                                                                                      initialization='random', max_iter=100, trial=2, metric='l2_distance', type_=type_, verbos=True)
elif ExpOnEmb == None:
    clusteringLabels['Raw'], clusteringNMI['Raw'], clusteringARI['Raw'], centers['Raw'] = utls.COPKmeansClusteringDBA(data, labels, n_clusters, ml, cl, 
                                                                                                      initialization='random', max_iter=100, trial=2, metric='dtw_distance', type_=type_, verbos=True)
else:
    clusteringLabels['Emb'], clusteringNMI['Emb'], clusteringARI['Emb'], centers['Emb'] = utls.COPKmeansClusteringDBA(embeddings.reshape(n,t,d), labels, n_clusters, [], [], 
                                                                                                      initialization='random', max_iter=100, trial=2, metric='l2_distance', type_=type_, verbos=True)
    clusteringLabels['Raw'], clusteringNMI['Raw'], clusteringARI['Raw'], centers['Raw'] = utls.COPKmeansClusteringDBA(data, labels, n_clusters, ml, cl, 
                                                                                                      initialization='random', max_iter=100, trial=2, metric='dtw_distance', type_=type_, verbos=True)
    
    
#%% Shapelet Assessment
S_T_dist = embeddings.T
shapelets = []
count=1
for i in range(len(model.shapelet_blocks)):
    for j in model.shapelet_blocks[i].weight.cpu().detach().numpy():
        print(count)
        shapelets.append(j[0])
        count+=1
shapelets = np.array(shapelets)
#%% Calculating the distance map between shapelets
ShapeletDistMapB = {}
for i in range(len(model.shapelet_blocks)):
    sblk = model.shapelet_blocks[i].weight.cpu().detach().numpy()[:,0,:]
    ShapeletDistMapB[i] = np.zeros([sblk.shape[0],sblk.shape[0]])
    for j in range(sblk.shape[0]):
        for jj in range(j,sblk.shape[0]):
            print(np.linalg.norm(sblk[j]-sblk[jj]))
            ShapeletDistMapB[i][j,jj] =np.round(np.linalg.norm(sblk[j]-sblk[jj]),0)
            # ShapeletDistMapB[i][j,jj] = np.corrcoef(sblk[j],sblk[jj])[0,1]
            # ShapeletDistMapB[i][j,jj] = utls.dtw_fast(sblk[j],sblk[jj])
            ShapeletDistMapB[i][jj,j] = ShapeletDistMapB[i][j,jj]
    plt.figure()
    plt.imshow(ShapeletDistMapB[i])
    plt.title(f'Block {i}')
#%% ordering the shapelet using different assessment approaches
OrderedShapelets = {}
if ExpOnEmb:
    for S in ['Local', 'Global']:
        for T in ['Single', 'Successive']:
            if T == 'Single':
                OrderedShapelets[f'{T}_{S}_Emb'] = qm.QualityMeasureIG(
                    S_T_dist, clusteringLabels['Emb'], Scope=S, Type=T, SingleShapelets=None, removepoints=True)
            else:
                OrderedShapelets[f'{T}_{S}_Emb'] = qm.QualityMeasureIG(
                    S_T_dist, clusteringLabels['Emb'], Scope=S, Type=T, SingleShapelets=OrderedShapelets[f'Single_{S}_Emb'], removepoints=True)
elif ExpOnEmb == None:
    for S in ['Local', 'Global']:
        for T in ['Single', 'Combined', 'Successive']:
            if T == 'Single':
                OrderedShapelets[f'{T}_{S}_Raw'] = qm.QualityMeasureIG(
                    S_T_dist, clusteringLabels['Emb'], Scope=S, Type=T, SingleShapelets=None, removepoints=True)
            else:
                OrderedShapelets[f'{T}_{S}_Raw'] = qm.QualityMeasureIG(
                    S_T_dist, clusteringLabels['Emb'], Scope=S, Type=T, SingleShapelets=OrderedShapelets[f'Single_{S}_Raw'], removepoints=True)

else:
    for testData in ['Emb', 'Raw']:
        for S in ['Local', 'Global']:
            for T in ['Single', 'Combined', 'Successive']:
                if T == 'Single':
                    OrderedShapelets[f'{T}_{S}_{testData}'] = qm.QualityMeasureIG(
                        S_T_dist, clusteringLabels[testData], Scope=S, Type=T, SingleShapelets=None, removepoints=True)
                else:
                    OrderedShapelets[f'{T}_{S}_{testData}'] = qm.QualityMeasureIG(
                        S_T_dist, clusteringLabels[testData], Scope=S, Type=T, SingleShapelets=OrderedShapelets[f'Single_{S}_{testData}'], removepoints=True)
    
#%% Plotting Best Shapelet according to the assessment and the corresponding clusters with the cloud of points using embedding distance

notallqm = True
qmtoplot = "Single"
scopetoplot = 'Local' #   plot both global and local if 'LocalGlobal' else scope
notallcluster = True
clustertoplot = 1
nmi = {}
ari = {}
shapeletstoplot=2
testData = 'Emb'
for QMunderstudy in [f"{T}_{S}_{testData}" for T in ["Single", "Combined", "Successive"] for S in ["Local", "Global"]]:  # need to add "Single"
    if notallqm:
        if qmtoplot not in QMunderstudy:
            continue
    qmplots.QMShapeletDistPerClusterPlot(QMunderstudy, OrderedShapelets[QMunderstudy],
                                         scopetoplot, notallcluster, clustertoplot, shapeletstoplot, embeddings, clusteringLabels[testData],
                                         labels, shapelets, dataset)


#%% UMAP plot of the Embedding
dirsave = None
algorithm = "CopkMeans"

notallqm = False
notallcluster = False
qmtoplot = "Single"
scopetoplot = 'Local' #   plot both global and local if 'LocalGlobal' else scope
shapeletstoplot=3
testData = 'Emb'

for QMunderstudy in [f"{T}_{S}_{testData}" for T in ["Single", "Successive"] for S in ["Local", "Global"]]:  # need to add "Single"

    if notallqm:
        if qmtoplot not in QMunderstudy:
            continue
    if 'Global'  in QMunderstudy and scopetoplot in QMunderstudy:
        sortedIG = OrderedShapelets[QMunderstudy].sort_values(by='IGMulti', ascending=False) if 'Single' in QMunderstudy else OrderedShapelets[QMunderstudy]
        shpToStudy = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudy else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
        # shpToStudy = shapelets[shapeletidx]     
        description = f"using Best {shapeletstoplot} Shapelets"
        print(f'{QMunderstudy} {description}, list of shapelets: {list(shpToStudy)}')
        if ExpOnEmb:
            labelType = 'Emb'
            utls. FeatureSpaceUMAP(embeddings[:,shpToStudy], dataset, np.array(clusteringLabels[labelType]), QMunderstudy, 
                                   alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                   description=description, algorithm=algorithm, dirsave=dirsave, palette='tab10',random_state=None)

    elif 'Local'  in QMunderstudy and scopetoplot in QMunderstudy:
        for g in OrderedShapelets[QMunderstudy].groupby(['Cluster']):
            sortedIG = g[1].sort_values(by='IG', ascending=False) if 'Single' in QMunderstudy else g[1] 
            if notallcluster:
                if int(sortedIG.Cluster.iloc[0]) != clustertoplot:
                        continue
            shpToStudy = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudy else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
            # shpToStudy = shapelets[shapeletidx] 
            description = f"using Best {shapeletstoplot} Shapelets w.r.t. cluster {sortedIG.Cluster.iloc[0]}"
            print(f'{QMunderstudy} {description}, list of shapelets: {list(shpToStudy)}')
            if ExpOnEmb:
                labelType = 'Emb'
                utls. FeatureSpaceUMAP(embeddings[:,shpToStudy], dataset, np.array(clusteringLabels[labelType]), QMunderstudy, 
                                       alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                       description=description, algorithm=algorithm, dirsave=dirsave, palette='tab10',random_state=None)
#%%
utls. FeatureSpaceUMAP(embeddings[:,:], dataset, np.array(labels), "ALLShapelets", 
                                       alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                      description="TrueLabels", algorithm=None, dirsave="None", palette='tab10',random_state=None)

utls. FeatureSpaceUMAP(embeddings[:,:], dataset, np.array(clusteringLabels['Emb']), "ALLShapelets", 
                                       alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                      description="clusteringLabels['Emb']", algorithm=None, dirsave="None", palette='tab10',random_state=None)
#%% PCA plot of the Embedding
dirsave = None
algorithm = "CopkMeans"

notallqm = True
notallcluster = False
qmtoplot = "Single"
scopetoplot = 'Local' #   plot both global and local if 'LocalGlobal' else scope
testData = 'Emb'

shapeletstoplot=10
n_components=3
for QMunderstudy in [f"{T}_{S}_{testData}" for T in ["Single", "Successive"] for S in ["Local", "Global"]]:  # need to add "Single"

    if notallqm:
        if qmtoplot not in QMunderstudy:
            continue
    if 'Global'  in QMunderstudy and scopetoplot in QMunderstudy:
        sortedIG = OrderedShapelets[QMunderstudy].sort_values(by='IGMulti', ascending=False) if 'Single' in QMunderstudy else OrderedShapelets[QMunderstudy]
        shpToStudy = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudy else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
        # shpToStudy = shapelets[shapeletidx]     
        description = f"using Best {shapeletstoplot} Shapelets"
        print(f'{QMunderstudy} {description}, list of shapelets: {list(shpToStudy)}')
        if ExpOnEmb:
            labelType = 'Emb'
            utls. FeatureSpacePCA(embeddings[:,shpToStudy], dataset, np.array(clusteringLabels[labelType]), QMunderstudy, 
                                   alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                   description=description, algorithm=algorithm, dirsave=dirsave, palette='tab10',n_components=n_components)

    elif 'Local'  in QMunderstudy and scopetoplot in QMunderstudy:
        for g in OrderedShapelets[QMunderstudy].groupby(['Cluster']):
            sortedIG = g[1].sort_values(by='IG', ascending=False) if 'Single' in QMunderstudy else g[1] 
            if notallcluster:
                if int(sortedIG.Cluster.iloc[0]) != clustertoplot:
                        continue
            shpToStudy = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudy else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
            # shpToStudy = shapelets[shapeletidx] 
            description = f"using Best {shapeletstoplot} Shapelets w.r.t. cluster {sortedIG.Cluster.iloc[0]}"
            print(f'{QMunderstudy} {description}, list of shapelets: {list(shpToStudy)}')
            if ExpOnEmb:
                labelType = 'Emb'
                utls. FeatureSpacePCA(embeddings[:,shpToStudy], dataset, np.array(clusteringLabels[labelType]), QMunderstudy, 
                                       alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                       description=description, algorithm=algorithm, dirsave=dirsave, palette='tab10',n_components=n_components)
#%%
nc=3
utls. FeatureSpacePCA(embeddings[:,:], dataset, np.array(labels), "ALLShapelets", 
                                   alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                   description="TrueLabels", algorithm=None, dirsave=None, palette='tab10',n_components=nc)

utls. FeatureSpacePCA(embeddings[:,:], dataset, np.array(clusteringLabels[labelType]), "ALLShapelets", 
                                   alpha=alpha, gamma=gamma, fr=fr, ml=ml, cl=cl, 
                                   description="clusteringLabels[labelType]", algorithm=None, dirsave=None, palette='tab10',n_components=nc)

#%% Plotting manifolds test
import skdim
from sklearn.manifold import MDS
#%%
danco = skdim.id.DANCo()
lpca = skdim.id.lPCA()
#%%
gid1pca = lpca.fit(embeddings).dimension_
gid2danco =danco.fit(embeddings).dimension_
print(gid1pca,gid2danco)
#%%
gid2dancol = danco.fit_pw(embeddings,n_neighbors=25).dimension_pw_
gid1pcal = lpca.fit_pw(embeddings,n_neighbors=25).dimension_pw_
#%%
#create a mish grid (surface) with the same coordinate as the points from the embeddings but with the hue of the gidpca and gid2danco
mesh = 1 
#%%
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(np.vstack([embeddings[:,:], centers['Emb'][:,:,0]]))
X_transformed = X_transformed[0:embeddings[:,:].shape[0],:]
#X_transfrom = embeddings[:, [4,5]]
Center_transformed = X_transformed[embeddings[:,:].shape[0]:,:]

#%%
fig  = plt.figure()
ax = fig.add_subplot()#projection='3d') #,X_transformed[:,2]
ax.scatter(X_transformed[:,0],X_transformed[:,1],c=[sns.color_palette('tab10',np.unique(np.array( clusteringLabels['Emb'])).shape[0]+1)[x] for x in list(np.array( clusteringLabels['Emb']).astype(int))],)
plt.title("MDS,  clusteringLabels['Emb']")
# ax.scatter(X_transformed[:,0],X_transformed[:,1],c=[sns.color_palette('tab10',np.unique(np.array(labels)).shape[0]+1)[x] for x in list(np.array(labels).astype(int))],)
# plt.title("Isomap, COLOR TRUE LABELS")

# ax.scatter(X_transformed[:,0],X_transformed[:,1],X_transformed[:,2],c=[sns.color_palette('tab10',np.unique(np.array(labels)).shape[0]+1)[x] for x in list(np.array(labels).astype(int))],)
# ax.scatter(X_transformed[:,0],X_transformed[:,1],c=[sns.color_palette('tab10',np.unique(np.array(clusteringLabels[labelType])).shape[0]+1)[x] for x in list(np.array(clusteringLabels[labelType]).astype(int))],)
#%% silhouette score
from sklearn.metrics import silhouette_samples,silhouette_score
silhouette_avg = silhouette_score(X_transformed, clusteringLabels['Emb'])#embeddings
print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
sample_silhouette_values = silhouette_samples(X_transformed, clusteringLabels['Emb'])
import SilhouetteStudy
SilhouetteStudy.silhouette_PlotAvgOrSample(dataset,X_transformed,n_clusters,clusteringLabels['Emb'],Center_transformed,fr,true_labels=labels,dimensions=(0,1),measureOndim=False)


# %%
