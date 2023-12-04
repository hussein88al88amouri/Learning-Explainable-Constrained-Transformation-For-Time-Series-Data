import sys
sys.path.append("/home/elamouri/CDPS/bin")
from QMplots import *
import warnings
from Utilities import *
import Quality_measures as qm
import seaborn as sns
import NN_CLDPS_Torch_Multi as CLDPSM
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import zscore
import argparse
warnings.filterwarnings("ignore")

def PCAreduced(data, n_components=2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

size=30

params = {#'legend.fontsize': 'large',
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.85,
          'ytick.labelsize': size*0.85,
          'axes.titlepad': 25
          }

parser = argparse.ArgumentParser(prog='Cluster Explanation', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('dataset', help='Dataset Name', type=str)
parser.add_argument('dpath', help='Dataset directory path', type=str)
parser.add_argument('Outdir', help="Absolute path to a directory to write the outputs to.\n", type=str)
parser.add_argument('--Alldata', dest='Alldata', help='Use test and train for training', default=False, action='store_true')
args = parser.parse_args()
print(args)

dpath = args.dpath #'/home/elamouri/Univariate_ts'#
dataset = args.dataset
Alldata = args.Alldata
outdirname  = 'SCE_TR' if Alldata else 'SCE_IN'
outdir =  args.Outdir + "/" + dataset + '/' + dataset + '_' + outdirname #'/home/elamouri/PatternRecognition/testpictures/SCE'
if outdir and not os.path.exists(outdir):
    os.makedirs(outdir)
    
print(f'Dataset: {dataset}')
Trdata, Trgt_labels = load_dataset_ts(dataset, 'TRAIN', dpath)
Tsdata, Tsgt_labels = load_dataset_ts(dataset, 'TEST', dpath)
n_clusters = n_classes(Trgt_labels)
print(f'The number of clusters is: {n_clusters}')
    
if Alldata:
    dataTm = np.concatenate((Trdata, Tsdata))
    dataTm = zscore(dataTm,axis=1)
    data = dataTm
    labels = np.concatenate((Trgt_labels, Tsgt_labels)).astype(np.int64)
    labelsTm = labels
    del Trdata, Trgt_labels, Tsdata, Tsgt_labels
else:
    dataTm = Trdata
    dataTm = zscore(dataTm,axis=1)
    labelsTm = Trgt_labels
    data = Tsdata
    data = zscore(data,axis=1)
    labels = Tsgt_labels.astype(np.int64)
    del Trdata, Tsdata, Tsgt_labels, Trgt_labels


fr_ = 0.25
ML, CL = constraint_generation_notarray(fr_, dataTm, labelsTm)
print('Initilized constraints')
DTWmax_ = DTWMax(dataTm)
lr = 0.01
lmin = 0.15
s = 3
rs = [lmin*i for i in range(1, s+1)]
r = 10
shapelet_lengths = {}
for sz in [int(p * dataTm.shape[1]) for p in rs]:
    n_shapelets = int(np.log(dataTm.shape[1] - sz) * r)
    shapelet_lengths[sz] = n_shapelets
citer = False
batch_size = 32
cinb = 8
alpha = 2.5
gamma = 2.5
epochs = 100
LDPS  = False
device = "cuda"
if LDPS:
    model = CLDPSM.CLDPSModel(n_shapelets_per_size=shapelet_lengths, lr=lr, fr=None, epochs=epochs,
                          batch_size=batch_size, ML=None, CL=None, gamma=None, alpha=None, constraints_in_batch=0,
                          device=device, saveloss=False, citer=citer, earlystoping=False, DTWmax=DTWmax_, type_='INDEP',)  # {30:5}
else:
    model = CLDPSM.CLDPSModel(n_shapelets_per_size=shapelet_lengths, lr=lr, fr=fr_, epochs=epochs,
                          batch_size=batch_size, ML=ML, CL=CL, gamma=gamma, alpha=alpha, constraints_in_batch=cinb,
                          device=device, saveloss=True, citer=citer, earlystoping=False, DTWmax=DTWmax_, type_='INDEP',)  # {30:5}

model._init_params(dataTm)


GEpoch = 5 if Alldata else 10
for i in range(GEpoch):
    model.fit(dataTm, init_=False)

# RKlabelsts, RKnmists, RKarists = KmeanClusteringTslearn(data, n_clusters, labels, verbos=True)

embTslearn = model._features(CLDPSM.tslearn2torch(data, device))
embTslearnNUmpy = embTslearn.cpu().detach(). numpy()
sdmat = SdmaT(sortdata(embTslearnNUmpy, labels)[0]) 

plt.figure()
plt.imshow(sortdata(embTslearnNUmpy, labels)[0], cmap='hot',aspect='auto')  # ;plt.imshow(embM)
plt.colorbar()
plt.title("Shapelet to time series distance map")
plt.savefig(f'{outdir}/{dataset}_shapelet_to_timeseries_distance_map.pdf')
plt.close('all')

CSKlabels, CSKnmis, CSKaris = KMeanClustering(embTslearnNUmpy, n_clusters, labels, verbos=True)

plotclusters(f'{dataset}_Train', data, CSKlabels, dir_=outdir, sample=True,samplenb=1)

S_T_dist = embTslearnNUmpy.T
shapelets_df = []
count = 1
for i in tqdm(range(len(model.shapelet_blocks))):
    for j in model.shapelet_blocks[i].weight.cpu().detach().numpy():
        shapelets_df.append(j[0])
        count += 1
shapelets_df = np.array(shapelets_df)

# Flag = False
# IGscoresSingleLocal = qm.QualityMeasureIG(S_T_dist, CSKlabels, Scope="Local", Type="Single", SingleShapelets=None, removepoints=Flag)
# IGscoresCombinedLocal = qm.QualityMeasureIG(S_T_dist, CSKlabels, Scope="Local", Type="Combined", SingleShapelets=IGscoresSingleLocal, removepoints=Flag)
# IGscoresSuccessiveLocal = qm.QualityMeasureIG(S_T_dist, CSKlabels, Scope="Local", Type="Successive", SingleShapelets=IGscoresSingleLocal, removepoints=Flag)
Flag = True
IGscoresSingleGlobal = qm.QualityMeasureIG(S_T_dist, CSKlabels, Scope="Global", Type="Single", SingleShapelets=None, removepoints=Flag)
# IGscoresCombinedGlobal = qm.QualityMeasureIG(S_T_dist, CSKlabels, Scope="Global", Type="Combined", SingleShapelets=IGscoresSingleGlobal, removepoints=Flag)
IGscoresSuccessiveGlobal = qm.QualityMeasureIG(S_T_dist, CSKlabels, Scope="Global", Type="Successive", SingleShapelets=IGscoresSingleGlobal, removepoints=Flag)

plt.rcParams.update(params)

notallqm = True
scopetoplot = 'Global'  
notallqm = True
notallcluster = False
clustertoplot = 1
shapeletstoplot = 3
for qmtoplot in ["Single", "Successive"]: # , "Combined"

    print(qmtoplot+scopetoplot)
    outdir_ = outdir + '/' +  qmtoplot + scopetoplot
    if outdir_ and not os.path.exists(outdir_):
        os.makedirs(outdir_)
    QMunderstudyName = f'IGscores{qmtoplot}{scopetoplot}'
    QMAHistplot(QMunderstudyName, vars()[QMunderstudyName], scopetoplot,dataset,outdir=outdir_,figsize=(10,7),lw=7)
    QMShapeletOrderline(QMunderstudyName, vars()[QMunderstudyName], scopetoplot, notallcluster,
                            clustertoplot, shapeletstoplot, embTslearnNUmpy, CSKlabels, labels, shapelets_df, dataset,outdir=outdir_,figsize=(7,5),lw=7,lp=10,s=200)
    QMunderstudy = vars()[QMunderstudyName]

    if 'Local' in QMunderstudyName:
        sortedIG = QMunderstudy.sort_values(by='IG', ascending=False) if 'Single' in QMunderstudyName else QMunderstudy
    elif 'Global' in QMunderstudyName:
        sortedIG = QMunderstudy.sort_values(by='IGMulti', ascending=False) if 'Single' in QMunderstudyName else QMunderstudy
    ig = sortedIG.IG.to_numpy() if 'Local' in QMunderstudyName else sortedIG.IGMulti.to_numpy()
    dataToplot = pd.concat([pd.DataFrame(data=embTslearnNUmpy, columns=range(embTslearnNUmpy.shape[1])), pd.DataFrame({"Predlabels": CSKlabels, "Truelabels": labels})], axis=1)
    dataToplot.reset_index(level=0, inplace=True)

    numshapelets = sortedIG.Shapelet.shape[0]
    pcaemb = PCAreduced(embTslearnNUmpy, n_components=numshapelets)

    CSKnmisA, CSKarisA = {}, {}
    CSKnmisAT, CSKarisAT = {}, {}
    PCA_CSKnmis, PCA_CSKaris = {}, {}
    PCA_CSKnmisT, PCA_CSKarisT = {}, {}
    for i in tqdm(range(sortedIG.Shapelet.shape[0])):
        shapeletidx = sortedIG.Shapelet.iloc[:i+1].to_numpy().astype(np.int64) if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[i].astype(np.int64)
        _, CSKnmisA[i], CSKarisA[i] = KMeanClustering(embTslearnNUmpy.T[shapeletidx].T, n_clusters, CSKlabels, verbos=False)
        _, CSKnmisAT[i], CSKarisAT[i] = KMeanClustering(embTslearnNUmpy.T[shapeletidx].T, n_clusters, labels, verbos=False)
        _, PCA_CSKnmisT[i], PCA_CSKarisT[i] = KMeanClustering(pcaemb[:,:i+1], n_clusters, labels, verbos=False)
        _, PCA_CSKnmis[i], PCA_CSKaris[i] = KMeanClustering(pcaemb[:,:i+1], n_clusters, CSKlabels, verbos=False)

    CIG = [np.sum(ig[:i]) for i in range(1,ig.shape[0]+1)]
    CIG = (CIG )/(np.max(CIG)) 
    IGvNMI = pd.DataFrame({"d":np.arange(1,sortedIG.shape[0]+1), 'IG':ig,'CIG':CIG, 'NMI%Pred':[np.mean(i) for i in CSKnmisA.values()]})
    fig = plt.figure(figsize=(10, 7), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.axhline(1, c='k', ls='dotted', linewidth=5)
    ax.axhline(0.8, c='k', ls='dotted', linewidth=5)
    sns.lineplot(data=IGvNMI, x="d", y='CIG', ax=ax,label='NCIG', linewidth=7)
    sns.lineplot(data=IGvNMI, x="d", y='NMI%Pred', ax=ax,label="SCE", linewidth=7)
    # sns.lineplot(data=IGvNMI, x="d", y='NMI%True', ax=ax,label="NMI%True", linewidth=7)
    plt.plot(np.arange(1,sortedIG.shape[0]+1), [np.mean(i) for i in PCA_CSKnmis.values()], label='PCA', linewidth=7)
    ax.set_ylabel("")
    ax.set_xlabel("number of shapelets")
    ax.set_ylim(0,1.05)
    ax.set_title(f'{QMunderstudyName} {dataset}')
    plt.legend( prop={'size': size})
    fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_IG_NCIG_NMI.pdf', bbox_inches='tight', pad_inches = 0)
    plt.close('all')
    
indexOf_Array_BTW_x_y = lambda Array, x, y: list(set(np.where(Array<x)[0]).intersection(np.where(Array>y)[0])) #list containing the first index BTW x,y
NumberofShapelets = lambda x: x+1
percOfShapelets  = lambda x: (x/embTslearnNUmpy.shape[1])*100
dtlv = lambda x: list(x.values()) # return the values of a dict as list
filename = 'ShapeletSelection_KMean_NMIscore.csv'
with open(f'{outdir}/{filename}', 'w') as f:
    txt = 'Dataset,NumberOfShapelets,PercOfShapelet,CIG,\
        NMI%Pred_mean,NMI%Pred_std,NMI%True_mean,NMI%True_std,\
        ARI%Pred_mean,ARI%Pred_std,ARI%True_mean,ARI%True_std,\
        NMI_FRun_mean,NMI_FRun_std,ARI_FRun_mean,ARI_FRun_std,\
        NMI_PCA_mean,NMI_PCA_std,NMI_PCA_meanT,NMI_PCA_stdT,\
        NMI_PCA_meanT,NMI_PCA_stdT,ARI_PCA_meanT,ARI_PCA_stdT\
        \n'
    txt += f'{dataset},{embTslearnNUmpy.shape[1]},{100},{CIG[-1]},\
        {np.mean(dtlv(CSKnmisA)[-1])},{np.std(dtlv(CSKnmisA)[-1])},{np.mean(dtlv(CSKnmisAT)[-1])},{np.std(dtlv(CSKnmisAT)[-1])},\
        {np.mean(dtlv(CSKarisA)[-1])},{np.std(dtlv(CSKarisA)[-1])},{np.mean(dtlv(CSKarisAT)[-1])},{np.std(dtlv(CSKarisAT)[-1])},\
        {np.mean(CSKnmis)},{np.std(CSKnmis)},{np.mean(CSKaris)},{np.std(CSKaris)},\
        {np.mean(dtlv(PCA_CSKnmis)[-1])},{np.std(dtlv(PCA_CSKnmis)[-1])},{np.mean(dtlv(PCA_CSKnmisT)[-1])},{np.std(dtlv(PCA_CSKnmisT)[-1])},\
        {np.mean(dtlv(PCA_CSKaris)[-1])},{np.std(dtlv(PCA_CSKaris)[-1])},{np.mean(dtlv(PCA_CSKarisT)[-1])},{np.std(dtlv(PCA_CSKarisT)[-1])}\
        \n'
    x, y = 10, 0.8
    idxOfCIG_xy = indexOf_Array_BTW_x_y(CIG,x,y)
    idxOfCIG_xy = idxOfCIG_xy[0]
    nOs = NumberofShapelets(idxOfCIG_xy)
    pOs = percOfShapelets(nOs)
    cig = CIG[idxOfCIG_xy]
    txt += f'{dataset},{nOs},{pOs},{cig},\
        {np.mean(dtlv(CSKnmisA)[idxOfCIG_xy])},{np.std(dtlv(CSKnmisA)[idxOfCIG_xy])},{np.mean(dtlv(CSKnmisAT)[idxOfCIG_xy])},{np.std(dtlv(CSKnmisAT)[idxOfCIG_xy])},\
        {np.mean(dtlv(CSKnmisA)[idxOfCIG_xy])},{np.std(dtlv(CSKnmisA)[idxOfCIG_xy])},{np.mean(dtlv(CSKnmisAT)[idxOfCIG_xy])},{np.std(dtlv(CSKnmisAT)[idxOfCIG_xy])},\
        {np.mean(CSKnmis)},{np.std(CSKnmis)},{np.mean(CSKaris)},{np.std(CSKaris)},\
        {np.mean(dtlv(PCA_CSKnmis)[idxOfCIG_xy])},{np.std(dtlv(PCA_CSKnmis)[idxOfCIG_xy])},{np.mean(dtlv(PCA_CSKnmisT)[idxOfCIG_xy])},{np.std(dtlv(PCA_CSKnmisT)[idxOfCIG_xy])},\
        {np.mean(dtlv(PCA_CSKaris)[idxOfCIG_xy])},{np.std(dtlv(PCA_CSKaris)[idxOfCIG_xy])},{np.mean(dtlv(PCA_CSKarisT)[idxOfCIG_xy])},{np.std(dtlv(PCA_CSKarisT)[idxOfCIG_xy])}\
        \n'
    f.write(txt)

# constraints = np.concatenate([ML,CL]).flatten()
# notconstraints = list(set(np.arange(0,data.shape[0])) - set(constraints))
# ddM = FDTWd(sortdata(data[notconstraints], labels[notconstraints])[0])
# sdmat_FE = SdmaT(sortdata(embTslearnNUmpy[notconstraints], labels[notconstraints])[0]) 
# sdmat_PE = SdmaT(sortdata(embTslearnNUmpy[notconstraints].T[IGscoresSuccessiveGlobal.Shapelet.iloc[idxOfCIG_xy]].T, labels[notconstraints])[0]) 
# uptr =  np.triu_indices(len(notconstraints))
# ddM_upr = ddM[uptr]
# sdmat_upr_FE = sdmat_FE[uptr]
# sdmat_upr_PE = sdmat_PE[uptr]
# df = pd.DataFrame({'Dataset':[dataset]*uptr[0].shape[0],'idx1':uptr[0],'idx2':uptr[1],'DTW_d':ddM_upr, 'DTW_a_FE':sdmat_upr_FE,'DTW_a_PE':sdmat_upr_PE})

# filename = 'sample_DTW_distance_comaprision.csv'
# if os.path.exists(f'{outdir}/{filename}'):
#     os.remove(f'{outdir}/{filename}')

# df.to_csv(f'{outdir}/{filename}',index=False) 

# filename = 'average_DTW_distance_comaprision.csv'
# if os.path.exists(f'{outdir}/{filename}'):
#     os.remove(f'{outdir}/{filename}')
# with open(f'{outdir}/{filename}','a') as f:
#     txt = 'Dataset,aDTW_d,aDTW_a_FE,DaTW_a_PE\n'
#     txt += f'{dataset},{np.average(ddM_upr)},{np.average(sdmat_upr_FE)},{np.average(sdmat_upr_PE)}'
#     f.write(txt)

#%%
