# %%  Adding bin to search path
import os
import sys
sys.path.append("../bin/")
from Utilities import *
sys.path.append("../CDPS/")
import CDPS_model as CDPS

import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import QMplots as qmplts 
import warnings
import time
import Quality_measures as qm
import random
import pandas as pd
import numpy as np
from scipy.stats import zscore
from glob import glob

import importlib
# %%
warnings.filterwarnings("ignore")
size=30
params = {#'legend.fontsize': 'large',
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.85,
          'ytick.labelsize': size*0.85,
          'axes.titlepad': 25
          }
plt.rcParams.update(params)


outdir__  
dpath = 
datasets = 
model = 

Trdata, Trgt_labels = load_dataset_ts(dataset, 'TRAIN', dpath)
Tsdata, Tsgt_labels = load_dataset_ts(dataset, 'TEST', dpath)
n_clusters = n_classes(Trgt_labels)

Alldata = True
if Alldata:
    dataTm = np.concatenate((Trdata, Tsdata))
    data = dataTm
    labels = np.concatenate((Trgt_labels, Tsgt_labels)).astype(np.int64)
    labelsTm = labels
    del Trdata, Trgt_labels, Tsdata, Tsgt_labels
else:
    dataTm = Trdata
    labelsTm = Trgt_labels
    data = Tsdata
    labels = Tsgt_labels.astype(np.int64)
    del Trdata, Tsdata, Tsgt_labels, Trgt_labels
dataTm = zscore(dataTm, axis=1)
data = zscore(data, axis=1)


RKlabelsts, RKnmists, RKarists = KmeanClusteringTslearn(
    data, n_clusters, labels, verbos=True)

embTslearn = model._features(CLDPSM.tslearn2torch(data, 'cpu'))
embTslearnNUmpy = embTslearn.cpu().detach(). numpy()
sdmat = SdmaT(sortdata(embTslearnNUmpy, labels)[0]) 



CSKlabels, CSKnmis, CSKaris = KMeanClustering(embTslearnNUmpy, n_clusters, labels, verbos=True)

S_T_dist = embTslearnNUmpy.T
shapelets_df = []
count = 1
for i in tqdm(range(len(model.shapelet_blocks))):
    for j in model.shapelet_blocks[i].weight.cpu().detach().numpy():
        shapelets_df.append(j[0])
        count += 1
shapelets_df = np.array(shapelets_df)
#%%
importlib.reload(qm)
plotclusters(f'{dataset}_Train:{data.shape[0]}_ClusterLabels',data,CSKlabels, dir_=outdir__, sample=True,samplenb=2)

# %% Local ranking by IG
Flag = False
IGscoresSingleLocal = qm.QualityMeasureIG(
    S_T_dist, CSKlabels, Scope="Local", Type="Single", SingleShapelets=None, removepoints=Flag)
#%%
IGscoresCombinedLocal = qm.QualityMeasureIG(
    S_T_dist, CSKlabels, Scope="Local", Type="Combined", SingleShapelets=IGscoresSingleLocal, removepoints=Flag)
IGscoresSuccessiveLocal = qm.QualityMeasureIG(
    S_T_dist, RKlabelsts, Scope="Local", Type="Successive", SingleShapelets=IGscoresSingleLocal, removepoints=Flag)
# %% Global ranking by IG
Flag = True
IGscoresSingleGlobal = qm.QualityMeasureIG(
    S_T_dist, CSKlabels, Scope="Global", Type="Single", SingleShapelets=None, removepoints=Flag)
#%%
IGscoresCombinedGlobal = qm.QualityMeasureIG(
    S_T_dist, CSKlabels, Scope="Global", Type="Combined", SingleShapelets=IGscoresSingleGlobal, removepoints=Flag)
#%%
IGscoresSuccessiveGlobal = qm.QualityMeasureIG(
    S_T_dist, CSKlabels, Scope="Global", Type="Successive", SingleShapelets=IGscoresSingleGlobal, removepoints=Flag)

# %% Plot of intra distance clusters for each shapelet
notallqm = False
qmtoplot = "Successive"
scopetoplot = 'Local'  # plot both global and local if 'LocalGlobal' else scope
plotSinglebycluster = False
clustertoplot = 3
nmi = {}
ari = {}
shapeletstoplot = 5
for QMunderstudy in ["IGscores"+i+j for i in ["Single", "Combined", "Successive"] for j in ["Local", "Global"]]:  # need to add "Single"
    if notallqm:
        if qmtoplot not in QMunderstudy:
            continue
        if scopetoplot not in QMunderstudy:
            print('Here')
            continue
    QMDistTSCplot(QMunderstudy, vars()[QMunderstudy], notallqm, qmtoplot, scopetoplot, plotSinglebycluster,
                  clustertoplot, shapeletstoplot, nmi, ari, embTslearnNUmpy, RKlabelsts, labels, n_clusters, dataset)
# %% Plot best shapelet and intra cluster distance for each ts and shapelet
notallqm = True
qmtoplot = "Single"
scopetoplot = 'Global'  # plot both global and local if 'LocalGlobal' else scope
notallcluster = False
clustertoplot = 1
nmi = {}
ari = {}
shapeletstoplot = 5
for QMunderstudy in ["IGscores"+i+j for i in ["Single", "Combined", "Successive"] for j in ["Local", "Global"]]:  # need to add "Single"
    if notallqm:
        if qmtoplot not in QMunderstudy:
            continue
    QMShapeletDistPerClusterPlot(QMunderstudy, vars()[QMunderstudy], scopetoplot, notallcluster,
                                 clustertoplot, shapeletstoplot, embTslearnNUmpy, CSKlabels, labels, shapelets_df, dataset)

 # %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def LDAonShapeletSpace(distvector,datapartition):
    datapartition = pd.Series(datapartition)
    lda_model = LinearDiscriminantAnalysis()
    classes =  np.unique(datapartition) if np.unique(datapartition).shape[0] > 2 else np.array([0])
    nclasses = np.unique(classes).shape[0]
    while True:
        try:
            lda_fit = lda_model.fit(distvector, datapartition)
        except:
            print("error trying again")
            continue
        break
    lda_relativedistances = lda_fit.decision_function(distvector)
    return lda_model.score(distvector, datapartition)

# %%
from sklearn.cluster import DBSCAN
def dbscan(data, truelabels):
    nmis = []
    aris = []
    
    db =  DBSCAN(eps=4, min_samples=10).fit(data)
    pred_labels = db.labels_
    nmis.append(normalized_mutual_info_score(truelabels, pred_labels))
    aris.append(adjusted_rand_score(truelabels, pred_labels))
    
    return pred_labels, nmis, aris

# %% Hist plot of shapelets IG
import QMplots as qmplts
importlib.reload(qmplts)
# %%
cnt=1

#%%
cnt = cnt+1

#%%
def PCAreduced(data, n_components=2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data
#%%
importlib.reload(qmplts)
# dataset=f'{datasets}'
outdirnameversion = '_ToPaper'
outdirname  = 'SCE_TR' if Alldata else 'SCE_IN'
notallqm = False
qmtoplot = "Combined"
scopetoplot = 'Local'  # plot both global and local if 'LocalGlobal' else scope

outdir = '/home/elamouri/PatternRecognition/testpictures/SCE/'+ dataset + outdirname + scopetoplot  + outdirnameversion
outdir = outdir__ + f'version_{cnt}' + qmtoplot
outdir = '/home/elamouri/Thesis/Figures/Chapter4/SCEreduction_Eval_10'
outdir = None
if outdir and not os.path.exists(outdir):
    os.makedirs(outdir)

plotclusters(f'{dataset}_Train:{data.shape[0]}_v2',data,CSKlabels, dir_=outdir, sample=True,samplenb=2)


for QMunderstudy in ["IGscores"+i+j for i in ["Single", "Combined", "Successive"] for j in [ "Local"]]:#"Local"
    if notallqm:
        if qmtoplot not in QMunderstudy:
            continue
    qmplts.QMAHistplot(QMunderstudy, vars()[QMunderstudy], scopetoplot,dataset,outdir=outdir,figsize=(10,7),lw=7)
    if outdir:
        vars()[QMunderstudy].to_csv(f'{outdir}/{QMunderstudy}.csv')

### %%
notallqm = True
# qmtoplot = "Combined"
# scopetoplot = 'Global'  # plot both global a+nd local if 'LocalGlobal' else scope
notallcluster = False
clustertoplot = 1
nmi = {}
ari = {}
shapeletstoplot = 3
for QMunderstudy in ["IGscores"+i+j for i in ["Single", "Combined", "Successive"] for j in ["Local", "Global"]]:  # need to add "Single"
    if notallqm:
        if qmtoplot+scopetoplot not in QMunderstudy:
            continue 
    print(qmtoplot+scopetoplot)
    qmplts.QMShapeletOrderline(QMunderstudy, vars()[QMunderstudy], scopetoplot, notallcluster,
                        clustertoplot, shapeletstoplot, embTslearnNUmpy, CSKlabels, labels, shapelets_df, dataset,outdir=outdir,figsize=(7,5),lw=7,lp=10,s=200)
##%%
shapeletstoplot = S_T_dist.shape[0]
QMunderstudyName = f'IGscores{qmtoplot}{scopetoplot}'
QMunderstudy = vars()[QMunderstudyName]
if 'Local' in QMunderstudyName:
    sortedIG = QMunderstudy.sort_values(by='IG', ascending=False) if 'Single' in QMunderstudyName else QMunderstudy
elif 'Global' in QMunderstudyName:
    sortedIG = QMunderstudy.sort_values(by='IGMulti', ascending=False) if 'Single' in QMunderstudyName else QMunderstudy
ig = sortedIG.IG.to_numpy() if 'Local' in QMunderstudyName else sortedIG.IGMulti.to_numpy()
# dataToplot = pd.concat([pd.DataFrame(data=embTslearnNUmpy, columns=range(embTslearnNUmpy.shape[1])), pd.DataFrame({"Predlabels": CSKlabels, "Truelabels": labels})], axis=1)
# dataToplot.reset_index(level=0, inplace=True)

numshapelets = sortedIG.Shapelet.shape[0]
PCAemb = PCAreduced(embTslearnNUmpy, n_components=numshapelets)


CSKlabelsA, CSKnmisA, CSKarisA, LDAscore = {}, {}, {}, {}
CSKlabelsAT, CSKnmisAT, CSKarisAT = {}, {}, {}
dbl, dbnmi, dbari = {}, {}, {}
PCAnmi = {}
for i in tqdm(range(sortedIG.Shapelet.shape[0])):
    shapeletidx = sortedIG.Shapelet.iloc[:i+1].to_numpy().astype(np.int64) if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[i].astype(np.int64)
    # dbl[i], dbnmi[i], dbari[i] = dbscan(embTslearnNUmpy.T[shapeletidx].T, CSKlabels)
    CSKlabelsA[i], CSKnmisA[i], CSKarisA[i] = KMeanClustering(embTslearnNUmpy.T[shapeletidx].T, n_clusters, CSKlabels, verbos=False)
    # CSKlabelsAT[i], CSKnmisAT[i], CSKarisAT[i] = KMeanClustering(embTslearnNUmpy.T[shapeletidx].T, n_clusters, labels, verbos=False)
    # LDAscore[i] = LDAonShapeletSpace(embTslearnNUmpy.T[shapeletidx].T,CSKlabels)
    _, PCAnmi[i], _ = KMeanClustering(PCAemb[:,:i+1], n_clusters, CSKlabels, verbos=False)

#"%% CHANGE TO WORK WITH CLUSTER WISE C
CIG = [np.sum(ig[:i]) for i in range(1,ig.shape[0]+1)]
CIG = (CIG )/(np.max(CIG)) 
# CLDA = [np.sum(list(LDAscore.values())[:i]) for i in range(1,len(LDAscore)+1)]
# CLDA = CLDA/np.max(CLDA)
IGvNMI = pd.DataFrame({"d":np.arange(1,sortedIG.shape[0]+1), 'IG':ig,'CIG':CIG, 'NMI%Pred':[np.mean(i) for i in CSKnmisA.values()]})
fig = plt.figure(figsize=(10, 7), tight_layout=True)
ax = fig.add_subplot(111)
ax.axhline(1, c='k', ls='dotted', linewidth=5)
ax.axhline(0.8, c='k', ls='dotted', linewidth=5)
# sns.lineplot(data=IGvNMI, x="d", y='IG', ax=ax,label="IG")
# ax.hist(IGvNMI.d, weights=IGvNMI.IG, bins=IGvNMI.shape[0], label="IG")
sns.lineplot(data=IGvNMI, x="d", y='CIG', ax=ax,label='NCIG', linewidth=7)
sns.lineplot(data=IGvNMI, x="d", y='NMI%Pred', ax=ax,label="NMI%Pred", linewidth=7)
# sns.lineplot(data=IGvNMI, x="d", y='NMI%True', ax=ax,label="NMI%True", linewidth=7)
# sns.lineplot(data=IGvNMI, x="d", y='LDAscore', ax=ax,label="LDAscore", linewidth=7)
# sns.lineplot(data=IGvNMI, x="d", y='CLDA', ax=ax,label="CLDA", linewidth=7)
# sns.lineplot(data=IGvNMI, x="d", y='dbscan', ax=ax,label="dbscan", linewidth=7)
# ax.axhline(np.mean(RKnmists), c='k', ls='--', label='RawNMI', linewidth=5)
ax.set_ylabel("")
ax.set_xlabel("number of shapelets")
ax.set_title(f'{QMunderstudyName} {dataset}')
plt.legend( prop={'size': size})
plt.ylim(0,1.05)
plt.figure()
##%%
if outdir:
    fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_IG_NCIG_NMI.pdf', bbox_inches='tight', pad_inches = 0)


plt.figure(figsize=(10, 7), tight_layout=True)
# sns.lineplot(data=IGvNMI, x="d", y='CIG',label='NCIG', linewidth=7)
sns.lineplot(data=IGvNMI, x="d", y='NMI%Pred', label="SCE", linewidth=7)
plt.plot(np.linspace(0,IGvNMI.shape[0],IGvNMI.shape[0]), [np.mean(i) for i in PCAnmi.values()], label='PCA',lw=7)
plt.xlabel("number of shapelets")
plt.ylabel("")
plt.ylim(0,1.05)
plt.legend( prop={'size': size})
if outdir:
    plt.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_PCA_NCIG_NMI.pdf', bbox_inches='tight', pad_inches = 0)
#%%
import matplotlib.colors as mcolors
colors = dict(zip(np.unique(CSKlabels),sns.color_palette("tab10", np.unique(CSKlabels).shape[0])))
listmarkers = ["o", "d", "X", "^", "D", "v", "<", ">"]
markersl = dict(zip(np.unique(labels), [listmarkers[i] for i in range(np.unique(labels).shape[0])]))
plt.figure(figsize=(7,5))
for label in set(labels):
    mask = [l == label for l in labels]
    plt.scatter([PCAemb[i,0] for i in range(PCAemb.shape[0]) if mask[i]] ,[PCAemb[i,1] for i in range(PCAemb.shape[0]) if mask[i]],c=[colors[i] for i in CSKlabels[mask]], marker=markersl[label],s=30)
plt.ylabel('PCA2')
plt.xlabel("PCA1")
plt.tight_layout()
if outdir:
    plt.savefig(f'{outdir}/PCA_2d_scatterplot.pdf')
#%%
# %%
idx=11
shapeletidxs = IGscoresSuccessiveGlobal.Shapelet.iloc[idx]
_ = FeatureSpaceUMAP(embTslearnNUmpy.T[shapeletidxs].T, dataset, labels, 'Truelabels')
# CSKlabels_, CSKnmis_, CSKaris_ = KMeanClustering(embTslearnNUmpy.T[shapeletidxs].T, n_clusters, labels, verbos=False)
_ = FeatureSpaceUMAP(embTslearnNUmpy.T[shapeletidxs].T, dataset, CSKlabels, 'CSKlabels_')
_ = FeatureSpaceUMAP(embTslearnNUmpy.T[shapeletidxs].T, dataset, CSKlabelsA[idx], 'CSKlabels_')


#%%
shp = 12
emb = embTslearnNUmpy.T[shp].T
df = pd.DataFrame({'emb':emb, 'CSKlabels':CSKlabels, 'labels':labels})
fig, ax = plt.subplots()
sns.kdeplot(data=df, x='emb',hue=CSKlabels,label='CSKlabels', ax=ax)
# sns.kdeplot(data=df, x='emb',label='Envelop',color="k",bw_adjust=0.4, ax=ax)
# ax.axvline(x=0.369)
 
# %%
from tslearn.barycenters import dtw_barycenter_averaging
fig = plt.figure(figsize=(10, 7), tight_layout=True)
classidx = 5
dd=0
timeseriesgrouped = data[labels==np.unique(labels)[classidx],:,0]
# timeseries = data[2,0:,0]
ax = fig.add_subplot(111)
lmt = 1 #timeseriesgrouped.shape[0]
# shpidx = IGscoresSuccessiveLocal[IGscoresSuccessiveLocal.Cluster == classidx].sort_values(by='IG',ascending=False).iloc[dd].Shapelet[-1]
# shpidx = 55
#IGscoresSuccessiveGlobal.iloc[dd].Shapelet[-1]
shapelet = shapelets_df[shpidx] #IGscoresSingleGlobal.sort_values(by='IGMulti',ascending=False)
shapelet = zscore(shapelet)
for i in range(0,lmt):
    timeseries = timeseriesgrouped[i+3]
    convs = np.correlate(timeseries.reshape((-1, )), shapelet.reshape((-1, )), mode="valid")
    idxmatch =  np.argmax(convs)
    # ax.plot(np.arange(0,timeseries.shape[0]),timeseries,lw=7)
xx = -10
lTS, lS = timeseries.shape[0] , shapelet.shape[0] 
padl, padr = idxmatch-xx , lTS - idxmatch -lS +xx
shapeletplot = np.pad(shapelet,(padl, padr),mode='constant',constant_values=(np.nan,))
ax.plot(np.arange(0,timeseries.shape[0]),shapeletplot,lw=7,color='#F97306')
ax.set_xlim(0,timeseries.shape[0])
#%%   Shapelet selection with a threshold of 80%
path = '/home/elamouri/PatternRecognition/pictures/SCE'
all_files = glob(os.path.join(path + "/*/*/ShapeletSelection*"), recursive=True)
li = []
for fn in all_files:
  if 'Coffee' not in fn and "Lightning2" not in fn and "BirdChicken" not in fn and "Meat" not in fn:
    df = pd.read_csv(fn, index_col=None, header=0)
    li.append(df)
scoresdf = pd.concat(li, axis=0, ignore_index=True)
scoresdf.rename({'        NMI%Pred_mean':'NMIPred_mean', 'NMI%Pred_std':'NMIPred_std', 'NMI%True_mean':'NMITrue_mean', 'NMI%True_std':'NMITrue_std','        ARI%Pred_mean':'ARIPred_mean', 'ARI%Pred_std':'ARIPred_std', 'ARI%True_mean':'ARITrue_mean', 'ARI%True_std':'ARITrue_std'},axis='columns',inplace=True)
scoresdf
# %% Write shapelet selection socres to latex
for df in scoresdf.groupby(by=['Dataset']):
  if df[1].shape[0] > 1:
      print(f"\multicolumn{{1}}{{c}}{{{df[0]}}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[0].CIG*100:.0f}\\%}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[1].CIG*100:.0f}\\%}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[0].NumberOfShapelets}}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[1].NumberOfShapelets}}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[0].NMIPred_mean:.2f}\\small{{\u00B1{df[1].iloc[0].NMIPred_std:.3f}}}}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[1].NMIPred_mean:.2f}\\small{{\u00B1{df[1].iloc[1].NMIPred_std:.3f}}}}} \\\ ")
    #   print(f"\multicolumn{{1}}{{c}}{{{df[0]}}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[1].NumberOfShapelets}({df[1].iloc[0].NumberOfShapelets})}} & \multicolumn{{1}}{{c}}{{{df[1].iloc[1].CIG*100:.0f}\\%}}  & \multicolumn{{1}}{{c}}{{{df[1].iloc[1].NMIPred_mean:.2f}\\small{{\u00B1{df[1].iloc[1].NMIPred_std:.3f}}}}} \\\ ")


#%% DTW comparision with approximation to shapelet selection subset approximat
path = '/home/elamouri/PatternRecognition/pictures/SCE'
all_files = glob(os.path.join(path + "/*/*/sample_DTW_distance_comaprision.csv"), recursive=True)
li = []
for fn in all_files:
  if 'Coffee' not in fn and "Lightning2" not in fn and "BirdChicken" not in fn and "Meat" not in fn:
    df = pd.read_csv(fn, index_col=None, header=0)
    li.append(df)
dtwdf = pd.concat(li, axis=0, ignore_index=True)
# %%
n, m = numSubplots(np.unique(dtwdf.Dataset).shape[0])
fig, axs = plt.subplots(n[0],n[1],figsize=(50,50))
c,r = 0,0
for df in dtwdf.groupby(by='Dataset'):
    sns.boxplot(data=df[1][["DTW_d","DTW_a_FE","DTW_a_PE"]],ax=axs[r][c])
    axs[r][c].set_title(f'{df[0]}')
    if c == n[1]-1:
        c = 0
        r +=1
    else:
        c += 1
#%%
for df in dtwdf.groupby(by='Dataset'):
    fig, axs = plt.subplots(1,3, figsize=(40, 40))
    msize = df[1].idx1.iloc[-1] +1
    DTW_d, DTW_a_FE, DTW_a_PE = np.zeros((msize,msize)), np.zeros((msize,msize)), np.zeros((msize,msize))
    DTW_d[[df[1].idx1,df[1].idx2]] = df[1].DTW_d
    DTW_a_FE[[df[1].idx1,df[1].idx2]] = df[1].DTW_a_FE
    DTW_a_PE[[df[1].idx1,df[1].idx2]] = df[1].DTW_a_PE
    DTW_a_FE = DTW_a_FE + DTW_a_FE.T
    DTW_a_PE = DTW_a_PE + DTW_a_PE.T
    DTW_d = DTW_d + DTW_d.T
    im0 = axs[0].imshow(DTW_d, cmap='hot')
    axs[0].set_title(f'{df[0]}_DTW')
    plt.colorbar(im0, fraction=0.046, ax=axs[0])
    im1 = axs[1].imshow(DTW_a_FE, cmap='hot',)
    axs[1].set_title(f'{df[0]}_DTW%FE')
    plt.colorbar(im1, fraction=0.046, ax=axs[1])
    im2 = axs[2].imshow(DTW_a_PE, cmap='hot')    
    axs[2].set_title(f'{df[0]}_DTW%PE')
    plt.colorbar(im2, fraction=0.046, ax=axs[2])
    fig.savefig(f'/home/elamouri/PatternRecognition/DTWComp/DTWCompv1/{df[0]}_DTW_distancemapComp.pdf')
# %%
indexOf_Array_BTW_x_y = lambda Array, x, y: list(set(np.where(Array<x)[0]).intersection(np.where(Array>y)[0])) #list containing the first index BTW x,y
NumberofShapelets = lambda x: x+1
percOfShapelets  = lambda x: (x/embTslearnNUmpy.shape[1])*100
dtlv = lambda x: list(x.values()) 
#%%
x, y = 10, 0.8
idxOfCIG_xy = indexOf_Array_BTW_x_y(CIG,x,y)
idxOfCIG_xy = idxOfCIG_xy[0]
nOs = NumberofShapelets(idxOfCIG_xy)
pOs = percOfShapelets(nOs)
cig = CIG[idxOfCIG_xy]
# %%
constraints = np.concatenate([ML,CL]).flatten()
notconstraints = list(set(np.arange(0,data.shape[0])) - set(constraints))
ddM = FDTWd(sortdata(data[notconstraints], labels[notconstraints])[0])
sdmat_FE = SdmaT(sortdata(embTslearnNUmpy[notconstraints], labels[notconstraints])[0]) 
sdmat_PE = SdmaT(sortdata(embTslearnNUmpy[notconstraints].T[IGscoresSuccessiveGlobal.Shapelet.iloc[idxOfCIG_xy]].T, labels[notconstraints])[0]) 
uptr =  np.triu_indices(len(notconstraints))
ddM_upr = ddM[uptr]
sdmat_upr_FE = sdmat_FE[uptr]
sdmat_upr_PE = sdmat_PE[uptr]
DTWComp = pd.DataFrame({'Dataset':[dataset]*uptr[0].shape[0],'idx1':uptr[0],'idx2':uptr[1],'DTW_d':ddM_upr, 'DTW_a_FE':sdmat_upr_FE,'DTW_a_PE':sdmat_upr_PE})

# %%
