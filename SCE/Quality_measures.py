#%%
from scipy.stats import entropy
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import special
from tqdm import tqdm

def QualityMeasureIG(Shapeletdists, Classlabels, Scope="Local", Type="Single",SingleShapelets=None,removepoints=True):
    
    
    if Scope == "Local":
        if Type == "Single":
           return  QMIGsingle(Shapeletdists, Classlabels)
        elif Type == "Combined":
            return  QMIGcombined(Shapeletdists, Classlabels,SingleShapelets)
        elif Type == "Successive":
            return  QMIGsuccsessive(Shapeletdists, Classlabels,SingleShapelets)
        
    elif Scope == "Global":
        COLUMN_NAMES = ['dim', 'Shapelet', 'IG']
        IGscores = pd.DataFrame(columns=COLUMN_NAMES)
        if Type == "Single":
            return QMIGsingleGlobal(Shapeletdists, Classlabels,removepoints=removepoints)
        elif Type == "Combined":
            return QMIGcombinedGlobal(Shapeletdists, Classlabels, SingleShapelets)
        elif Type == "Successive":
            return QMIGsuccsessiveGlobal(Shapeletdists, Classlabels, SingleShapelets)
   
def QMIGsuccsessiveGlobal(Shapeletdists, Classlabels, SingleShapelets):
    COLUMN_NAMES = ["dim",'Shapelet', 'IGMulti','Partitions']
    IGscoresMulti = pd.DataFrame(columns=COLUMN_NAMES)   
    labels_ = pd.Series(Classlabels)
    classes = np.unique(Classlabels) #if np.unique(Classlabels).shape[0] > 2 else np.array([0])
    SingleShapelets = SingleShapelets.sort_values('IGMulti',ascending=False)   
    sidx  = SingleShapelets.Shapelet.to_numpy()
    bidx = np.array([sidx[0]])
    maxIG = SingleShapelets.IGMulti.iloc[0]
    PPartition = SingleShapelets.Partitions.iloc[0]
    # PPartition = {}
    # for i in temp.keys():
    #     PPartition[dominantclass(temp[i])] = temp[i]
    # del temp  
    IGscoresMulti = IGscoresMulti.append(pd.Series({'dim':1, 'Shapelet': bidx , 'IGMulti': maxIG, 'Partitions':PPartition},name='dictIG'),ignore_index=True) 
    for d in tqdm(range(2,Shapeletdists.shape[0]+1)):    
        # if d == 1:
        # else:
            ndg = sidx.shape[0]-d+1
            cS_indx = np.empty((d,ndg))    
            cS_indx[:d-1] = np.vstack([bidx]*ndg).transpose((1,0))
            dsidx = sidx[~np.in1d(sidx,bidx.reshape((1,-1)))]
            dsidx = np.flip(dsidx)
            cS_indx[d-1:] = dsidx
            cS_indx = cS_indx.astype(np.int64)
            maxIG = 0
            for i in cS_indx.transpose():
                distvector = Shapeletdists[i].T
                _, CPartition, ldascore= IGmultipleGlobal(distvector, labels_)
                Dpartition, mv, inc, M, IN, correctclusterd = {}, {}, {}, {}, {}, {}
                for j in CPartition.keys() & PPartition.keys():
                    mv[j] = CPartition[j][~np.in1d(CPartition[j].index.to_numpy(),PPartition[j].index.to_numpy())]
                    inc[j] = CPartition[j][list(set(CPartition[j][CPartition[j] != j].index.to_numpy()) - set(mv[j].index.to_numpy()))]
                    # inc[j] = inc[j][list(set(inc[j].index.to_numpy()) - set(mv[j].index.to_numpy()))]
                    Dpartition[j]  = pd.concat([inc[j],mv[j]])
                    M[j] = mv[j][mv[j] == j]
                    IN[j] = mv[j][mv[j] != j]
                    correctclusterd[j] = CPartition[j][list(set(CPartition[j][CPartition[j] == j].index.to_numpy()) - set(mv[j].index.to_numpy()))]
                    # Dpartition[j] = CPartition[j][~np.in1d(CPartition[j].index.to_numpy(),PPartition[j].index.to_numpy())]
                M = np.sum([len(i) for i in M.values()])
                NN = np.sum([len(i) for i in IN.values()])
                mvl = np.sum([len(i) for i in Dpartition.values()])
                ninc  = np.sum([len(i) for i in inc.values()])
                nmbcorrectcluster = np.sum([len(i) for i in correctclusterd.values()])
                datapartitionNew = labels_[np.hstack([i.index for i in Dpartition.values()])]
                Probabilitypartitions = [i.value_counts(normalize=True) for i in Dpartition.values()]
                base = np.unique(labels_).shape[0]
                ID =  MClassEntropy(datapartitionNew.value_counts(normalize=True),base=base)
                IDafter = [(Dpartition[i].shape[0]/datapartitionNew.shape[0]) * MClassEntropy(Probabilitypartitions[j],base=base) for i,j in zip(Dpartition.keys(),range(len(Probabilitypartitions)))] if datapartitionNew.shape[0] != 0 else ID
                IGaftersplit = ID - np.sum(IDafter)#*(1/(M+1))
                IGaftersplit = IGaftersplit * M/labels_.shape[0] + 0.00000000000000000001
                if maxIG <= IGaftersplit:
                    bidx = i.transpose()
                    maxIG = IGaftersplit 
                    BPartition = CPartition
            PPartition  = BPartition
            IGscoresMulti = IGscoresMulti.append(pd.Series({'dim':d, 'Shapelet': bidx , 'IGMulti': maxIG, 'Partitions':BPartition},name='dictIG'),ignore_index=True) 
    return IGscoresMulti

def QMIGsuccsessive(Shapeletdists, Classlabels, SingleShapelets):
    # import pdb;pdb.set_trace();
    COLUMN_NAMES = ['dim', 'Shapelet', 'IG', 'Cluster', 'D1s', 'D2s', 'D1', 'D2']
    IGscores = pd.DataFrame(columns=COLUMN_NAMES)
    labels_ = pd.Series(Classlabels)
    classes = np.unique(Classlabels) #if np.unique(Classlabels).shape[0] > 2 else np.array([0])
    SingleShapelets = SingleShapelets.sort_values('IG',ascending=False)
    for l, li in zip(classes,range(classes.shape[0])):
        sidx = np.array([SingleShapelets[SingleShapelets.Cluster == l].Shapelet.astype(np.int64)])[0]
        datapartition = labels_.replace(to_replace=classes[np.arange(classes.shape[0])!=li], value=-1)     
        for d in range(1,Shapeletdists.shape[0]):    
            if d == 1:
                bidx = np.array([sidx[0]])
                distvector = pd.Series(Shapeletdists[bidx[0]]).sort_values()
                maxIG, _, D1smax, D2smax, PD1 , PD2 , = IG1D(distvector ,datapartition)
                BCD1 = PD1.to_numpy()
                BCD2 = PD2.to_numpy()
                IGaftersplit = maxIG
            else:
                ndg = sidx.shape[0]-d+1
                cS_indx = np.empty((d,ndg))    
                cS_indx[:d-1] = np.vstack([bidx]*ndg).transpose((1,0))
                cS_indx[d-1:] = sidx[~np.in1d(sidx,bidx.reshape((1,-1)))]
                cS_indx = cS_indx.astype(np.int64)
                maxIG = 0
                for i in cS_indx.transpose():
                    distvector = Shapeletdists[i].T
                    _, _, _, D1, D2 = IGmultiple(distvector, datapartition)
                    bidx = i.transpose()
                    CD1 = D1.to_numpy()
                    CD2 = D2.to_numpy()
                    D1n = CD1[~np.in1d(CD1,PD1)]
                    D2n = CD2[~np.in1d(CD2,PD2)]
                    datapartitionNew = datapartition[np.hstack([D2n,D1n])]
                    base = np.unique(datapartition).shape[0]
                    ID = entropy(datapartitionNew.value_counts(normalize=True),base=base) 
                    fD1 = D1n.shape[0]/datapartitionNew.shape[0] if datapartitionNew.shape[0] != 0 else 0
                    fD2 = D2n.shape[0]/datapartitionNew.shape[0] if datapartitionNew.shape[0] != 0 else 0
                    ID_hat = fD1 * entropy(datapartitionNew[D1n].value_counts(normalize=True),base=base) + fD2 * entropy(datapartitionNew[D2n].value_counts(normalize=True),base=base)
                    IGaftersplit = ID - ID_hat  
                    IGaftersplit = IGaftersplit * (datapartitionNew.shape[0] / datapartition.shape[0])
                    if maxIG < IGaftersplit:
                        D1smax = D1n.shape[0]
                        D2smax = D2n.shape[0]
                        bidx = i.transpose()
                        maxIG = IGaftersplit 
                        BCD1 = CD1
                        BCD2 = CD2
                PD1  = BCD1
                PD2 = BCD2
            IGscores = IGscores.append(pd.Series({'dim':d, 'Shapelet': bidx, 'IG':maxIG ,'Cluster':l, 'D1s':D1smax, 'D1s':D2smax ,'D1':PD1, 'D2': PD2},name='dictIG'),ignore_index=True) 
    return IGscores

def QMIGcombinedGlobal(Shapeletdists, Classlabels, SingleShapelets):
    COLUMN_NAMES = ["dim",'Shapelet', 'IGMulti','Partitions']
    IGscoresMulti = pd.DataFrame(columns=COLUMN_NAMES)   
    labels_ = pd.Series(Classlabels)
    SingleShapelets = SingleShapelets.sort_values('IGMulti',ascending=False)   
    for d in tqdm(range(2,Shapeletdists.shape[0])):    
            if d == 2:
                sidx = SingleShapelets.Shapelet.to_numpy()
                bidx = np.array([sidx[0]])
                IGscoresMulti = IGscoresMulti.append(pd.Series({'dim':1, 'Shapelet': bidx , 'IGMulti': SingleShapelets.IGMulti.iloc[0], 'Partitions':SingleShapelets.Partitions.iloc[0]},name='dictIG'),ignore_index=True)                
            ndg = sidx.shape[0]-d+1
            cS_indx = np.empty((d,ndg))    
            cS_indx[:d-1] = np.vstack([bidx]*ndg).transpose((1,0))
            cS_indx[d-1:] = sidx[~np.in1d(sidx,bidx.reshape((1,-1)))]
            cS_indx = cS_indx.astype(np.int64)
            maxIG = 0
            MPartitions = {}
            for i in cS_indx.transpose():
                distvector = np.transpose(Shapeletdists[i],(1,0))
                IGscore, Partitions, _= IGmultipleGlobal(distvector, labels_)
                if maxIG < IGscore:
                    maxIG = IGscore
                    MPartitions = Partitions
                    bidx = i.transpose()
            IGscoresMulti = IGscoresMulti.append(pd.Series({'dim':d, 'Shapelet': bidx , 'IGMulti': maxIG, 'Partitions':MPartitions},name='dictIG'),ignore_index=True) 
    return IGscoresMulti

def QMIGcombined(Shapeletdists, Classlabels, SingleShapelets):
    COLUMN_NAMES = ['dim', 'Shapelet', 'IG', 'Cluster','splitdist','D1s','D2s','D1','D2']
    IGscores = pd.DataFrame(columns=COLUMN_NAMES)
    labels_ = pd.Series(Classlabels)
    classes = np.unique(Classlabels)# if np.unique(Classlabels).shape[0] > 2 else np.array([0])
    SingleShapelets = SingleShapelets.sort_values('IG',ascending=False)   
    for l, li in zip(classes,range(classes.shape[0])):
        for d in range(2,Shapeletdists.shape[0]):    
            datapartition = labels_.replace(to_replace=classes[np.arange(classes.shape[0])!=li], value=-1)     
            if d == 2:
                sidx = np.array([SingleShapelets[SingleShapelets.Cluster == l].Shapelet.astype(np.int64)])[0]
                bidx = np.array([sidx[0]])
                IGscores= IGscores.append(pd.Series({'dim':1, 'Shapelet': bidx , 'IG':SingleShapelets[SingleShapelets.Cluster == l].IG.iloc[0] ,'Cluster':l, 'D1s':SingleShapelets[SingleShapelets.Cluster == l].D1s.iloc[0], 'D2s':SingleShapelets[SingleShapelets.Cluster == l].D2s.iloc[0], 'D1':SingleShapelets[SingleShapelets.Cluster == l].D1.iloc[0], 'D2':SingleShapelets[SingleShapelets.Cluster == l].D2.iloc[0]},name='dictIG'),ignore_index=True)                
            ndg = sidx.shape[0]-d+1
            cS_indx = np.empty((d,ndg))    
            cS_indx[:d-1] = np.vstack([bidx]*ndg).transpose((1,0))
            cS_indx[d-1:] = sidx[~np.in1d(sidx,bidx.reshape((1,-1)))]
            cS_indx = cS_indx.astype(np.int64)
            maxIG = 0
            for i in cS_indx.transpose():
                distvector = np.transpose(Shapeletdists[i],(1,0))
                IGscore, D1s, D2s, D1l, D2l= IGmultiple(distvector, datapartition)
                if maxIG < IGscore:
                    maxIG = IGscore
                    D1smax = D1s
                    D2smax = D2s
                    D1L = D1l
                    D2L = D2l
                    bidx = i.transpose()
            IGscores = IGscores.append(pd.Series({'dim':d, 'Shapelet': bidx, 'IG':maxIG ,'Cluster':l, 'D1s':D1smax, 'D2s': D2smax, 'D1':D1L, 'D2': D2L},name='dictIG'),ignore_index=True) 
    return IGscores


def QMIGsingleGlobal(Shapeletdists, Classlabels,removepoints=True):
    COLUMN_NAMES = ['dim','Shapelet', 'IGMulti','IGaverage','Partitions']
    IGscoresMulti = pd.DataFrame(columns=COLUMN_NAMES)
    labels_ = pd.Series(Classlabels)
    classes = np.unique(Classlabels) if np.unique(Classlabels).shape[0] > 2 else np.array([0])
    clustersizes = {}
    clustersidx  = {}
    clustersplits = {}
    clustersIG = {}
    for s, si in tqdm(zip(Shapeletdists,range(Shapeletdists.shape[0]))):
        distsort = pd.Series(s).sort_values()
        templabels = labels_
        tempdist = distsort
        for l, li in zip(classes,range(classes.shape[0])):
            datapartition = templabels.replace(to_replace=classes[np.arange(classes.shape[0])!=li], value=-1)
            IGscoreBYcluster, splitdistBYcluster, D1sBYcluster, D2sBYcluster, D1BYcluster, D2BYcluster = IG1D(tempdist ,datapartition)
            clustersIG[l], clustersizes[l], clustersidx[l], clustersplits[l] =IGscoreBYcluster, [D1sBYcluster,D2sBYcluster], [D1BYcluster,D2BYcluster], splitdistBYcluster
            if removepoints:
                tempdist = tempdist[templabels != l]
                templabels = templabels[templabels != l]
        base = np.unique(labels_).shape[0]
        ID = MClassEntropy(labels_.value_counts(normalize=True),base=base)
        Dpartitions  = RetrunPartitionsizes(labels_,distsort, np.sort(np.array(list(clustersplits.values()))),removepoints=removepoints)
        Probabilitypartitions = [i.value_counts(normalize=True) for i in Dpartitions.values()]
        IDafter = [(Dpartitions[i].shape[0]/distsort.shape[0]) * MClassEntropy(Probabilitypartitions[j],base=base) for i,j in zip(Dpartitions.keys(),range(len(Probabilitypartitions)))]
        IGmulticlass = ID - np.sum(IDafter)
        IGscoresMulti = IGscoresMulti.append(pd.Series({'dim':1,'Shapelet':si, 'IGaverage':np.mean([i for i in clustersIG.values()]),'IGMulti':IGmulticlass,'Partitions':Dpartitions}),ignore_index=True)
    return IGscoresMulti

def QMIGsingle(Shapeletdists, Classlabels):
    COLUMN_NAMES = ['dim', 'Shapelet', 'IG', 'Cluster','splitdist','D1s','D2s','D1','D2']
    IGscores = pd.DataFrame(columns=COLUMN_NAMES)
    labels_ = pd.Series(Classlabels)
    classes = np.unique(Classlabels) #if np.unique(Classlabels).shape[0] > 2 else np.array([0])
    for s, si in zip(Shapeletdists,range(Shapeletdists.shape[0])):
        distsort = pd.Series(s).sort_values()
        for l, li in zip(classes,range(classes.shape[0])):
            datapartition = labels_.replace(to_replace=classes[np.arange(classes.shape[0])!=li], value=-1)
            IGscore, splitdist, D1s, D2s, D1, D2 = IG1D(distsort ,datapartition)
            IGscores = IGscores.append(pd.Series({'dim':1,'Shapelet':si,'IG':IGscore ,'Cluster':l, "splitdist":splitdist, 'D1s':D1s, 'D2s':D2s, 'D1':D1, 'D2':D2},name='dictIG'),ignore_index=True)
    return IGscores

def IGmultipleGlobal(distvector,datapartition):
    lda_model = LinearDiscriminantAnalysis()
    classes =  np.unique(datapartition)
    while True:
        try:
            lda_fit = lda_model.fit(distvector, datapartition)
        except:
            print("error trying again")
            continue
        break
    lda_pred = pd.Series(lda_fit.predict(distvector))
    Bpartitions = {}
    base = np.unique(datapartition).shape[0]
    for c in classes:
         Bpartitions[c] = datapartition[lda_pred == c]
    Probabilitypartitions = [i.value_counts(normalize=True) for i in Bpartitions.values()]
    IDafter = [(Bpartitions[i].shape[0]/distvector.shape[0]) * MClassEntropy(Probabilitypartitions[j],base=base) for i,j in zip(Bpartitions.keys(),range(len(Probabilitypartitions)))]
    ID =  MClassEntropy(datapartition.value_counts(normalize=True),base=base)
    IG = ID - np.sum(IDafter)
    
    return IG, Bpartitions, lda_model.score(distvector, datapartition)

def IGmultiple(distvector,datapartition):
    base = np.unique(datapartition).shape[0]
    ID = entropy(datapartition.value_counts(normalize=True),base=base)
    lda_model = LinearDiscriminantAnalysis()
    while True:
        try:
            lda_fit = lda_model.fit(distvector, datapartition)
        except:
            print("error trying again")
            continue
        break
    lda_relativedistances = pd.Series(lda_fit.decision_function(distvector))
    D1 = datapartition[lda_relativedistances [lda_relativedistances <= 0].index]
    D2 = datapartition[lda_relativedistances [lda_relativedistances > 0].index]
    fD1 = D1.shape[0]/distvector.shape[0]
    fD2 = D2.shape[0]/distvector.shape[0]
    ID_hat = fD1 * entropy(D1.value_counts(normalize=True),base=base) + fD2 * entropy(D2.value_counts(normalize=True),base=base)
    IG = ID - ID_hat
    return IG, D1.shape[0], D2.shape[0], lda_relativedistances[lda_relativedistances <= 0].index, lda_relativedistances[lda_relativedistances > 0].index
    
def IG1D(dist_sort,datapartition):
    base = np.unique(datapartition).shape[0]
    dist_labels = datapartition[dist_sort.index]
    split_dists = np.unique(np.around(dist_sort.values,decimals=5))
    split_dists = np.mean(split_dists.reshape((-1,2)) if split_dists.shape[0]%2 == 0 else np.pad(split_dists, (0,1), 'constant', constant_values=split_dists[-1]).reshape((-1,2)), axis=1)
    ID = entropy(dist_labels.value_counts(normalize=True),base=base)
    maxIG = 0
    split_max = 0
    D1s = np.array([])
    D2s = np.array([])
    D1 = np.array([])
    D2 = np.array([])
    D1idx = np.array([])
    D2idx = np.array([])
    for split in split_dists:
        D1 = dist_labels[dist_sort[dist_sort <= split].index]
        D2 = dist_labels[dist_sort[dist_sort > split].index]
        fD1 = D1.shape[0]/dist_labels.shape[0]
        fD2 = D2.shape[0]/dist_labels.shape[0]
        ID_hat = fD1 * entropy(D1.value_counts(normalize=True),base=base) + fD2 * entropy(D2.value_counts(normalize=True),base=base)
        IGscore = ID - ID_hat
        if maxIG < IGscore:
            maxIG = IGscore
            split_max= split
            D1s = D1.shape[0]
            D2s = D2.shape[0]
            D1idx = dist_sort[dist_sort <= split].index
            D2idx = dist_sort[dist_sort > split].index
    return maxIG, split_max, D1s, D2s, D1idx, D2idx


def MClassEntropy(pk, axis=0,base=np.e):
     pk = np.asarray(pk)
     pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
     vec = special.entr(pk) #+ special.entr(1-pk)
     S = np.sum(vec/np.log(base), axis=axis)
     return S

def RetrunPartitionsizes(distlabels,distsort,splitsort,removepoints=False):
    splitsort = splitsort[1:] if splitsort[0] == 0 else splitsort
    Dpartitions = {}
    if np.unique(distlabels).shape[0] > 2:
        splitsort = getsplits(splitsort) if not removepoints else splitsort
        Dpartitions[0] = distlabels[distsort < splitsort[0]]
        Dpartitions[np.unique(distlabels).shape[0]-1] = distlabels[distsort > splitsort[-1]]
        for i in range(0,splitsort.shape[0]-1):
            Dpartitions[i+1] = distlabels[distsort.between(splitsort[i],splitsort[i+1], inclusive = True)]
    else:
        Dpartitions[0] = distlabels[distsort <= splitsort[0]]
        Dpartitions[1] = distlabels[distsort > splitsort[0]]
    # temp = {}
    # for members in Dpartitions.values():
    #     temp[dominantclass(members)] = members 
    # Dpartitions = temp
    # del temp
    return Dpartitions

def dominantclass(array):
    members = np.unique(array, return_counts=True)
    domclass = members[0][np.argmax(members[1])]
    return domclass

def getsplits(splits):
    dist = []
    for i in range(splits.shape[0]-1):
        dist.append(np.abs(splits[i]-splits[i+1]))
    nsplits = []
    if np.argmin(dist) == 0:
        nsplits = [np.mean([splits[0],splits[1]]),splits[2]]
    elif np.argmin(dist) == 1:
        nsplits = [splits[0],np.mean([splits[1],splits[2]])]
    return nsplits
# %%
