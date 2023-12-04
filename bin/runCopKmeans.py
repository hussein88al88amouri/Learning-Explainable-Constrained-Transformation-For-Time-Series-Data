import numpy as np
from glob import glob
import Utilities as utils
import argparse
from argparse import RawTextHelpFormatter
import os
import gc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CopKmeas Clustering',description='Data clustering using COP-Kmeans and DBA averaging', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('Outdir', help="Absolute path to a directory to write the outputs to.\n", type=str)
    parser.add_argument('DataName', help='Dataset Name', type=str)
    parser.add_argument('DataPath', help='Dataset DIR path', type=str)
    parser.add_argument('DataType', help='CDPSEmb or Raw data', type=str)
    parser.add_argument("--CDPSEmbPath", nargs="?", dest='CDPSEmbPath', default=None, type=str, help='path to the CDPS embedings')
    parser.add_argument("--fr", nargs="?", dest='fr', default=None, type=float, help='Constraint fraction from dataset')
    parser.add_argument("--Constraints", dest='Constraints', help="Constraints path dir",nargs="?", default=None, type=str)
    parser.add_argument('--nfr', help='Number of given constraint ratio set', type=int, default=None)
    parser.add_argument('--type', help='Type of DTW similarity to use: dependent or independent', type=str, default=None)
    parser.add_argument('--metric', help='l2_distance or dtw_distance', type=str, default='l2_distance')
    parser.add_argument('--trial', help='Number of times to repeat the clustering', type=int, default=10)
    parser.add_argument('--max_iter', help='Max number of iterations', type=int, default=100)
    parser.add_argument('--initialization', help='random or kmpp initialization', type=str, default='random')
    parser.add_argument('--trialNumber', help='Current test number', type=int, default=0)
    parser.add_argument('--Alldata', dest='Alldata', help='Use test and train for training', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    
    
    if args.DataType == 'Raw':
        Trdata, Trgt_labels = utils.load_dataset_ts(args.DataName,'TRAIN',args.DataPath)
        Tsdata, Tsgt_labels = utils.load_dataset_ts(args.DataName,'TEST',args.DataPath)
    
        print("Running on Raw Time Series")
        data = np.concatenate((Trdata,Tsdata))
        labels = np.concatenate((Trgt_labels,Tsgt_labels))
        labels = utils.labelsToint(labels).astype(float).astype(int)
        n_clusters = utils.n_classes(labels)
        del Trdata, Trgt_labels, Tsdata, Tsgt_labels
        gc.collect()
    
        if args.Alldata:
            alldata = "Alldata"
        else:
            alldata = "Traindata"
            
        if args.Constraints != None and args.fr != 0:
            print('Loading Constraints')
            ML = np.load(os.path.join(args.Constraints,args.DataName+ f'_{alldata}/{args.DataName}_{alldata}_fr_{args.fr:.2f}_ML_tn_{args.nfr}.npy'))
            CL = np.load(os.path.join(args.Constraints,args.DataName + f'_{alldata}/{args.DataName}_{alldata}_fr_{args.fr:.2f}_CL_tn_{args.nfr}.npy'))
        else:
                ML ,CL = [], []
                
        normr = lambda x: (x/np.sqrt(np.square(x).sum(axis=0)))
        datanorm = np.zeros(data.shape)
        for i in range(data.shape[0]):
            datanorm[i] = normr(data[i])
        data = datanorm
        del datanorm
        gc.collect()

    elif args.DataType == "CDPSEmb":
        print('Running on CDPS Embedings')
        _, Trgt_labels = utils.load_dataset_ts(args.DataName,'TRAIN',args.DataPath)
        _, Tsgt_labels = utils.load_dataset_ts(args.DataName,'TEST',args.DataPath)
        if args.Alldata:
            alldata = "Alldata"
            labels = np.concatenate((Trgt_labels,Tsgt_labels))
            labels = utils.labelsToint(labels).astype(float).astype(int)
            n_clusters = utils.n_classes(labels)
            del Trgt_labels, Tsgt_labels
        else:
            alldata = "Traindata"
            labels = Tsgt_labels
            labels = utils.labelsToint(labels).astype(float).astype(int)
            n_clusters = utils.n_classes(labels)
            del Trgt_labels, Tsgt_labels
        
        data = np.load(args.CDPSEmbPath)
        data = data.reshape((data.shape[0], data.shape[1], 1))
        ML, CL = [], []
                    
    metric = 'Euclidean' if args.metric == 'l2_distance' else 'DTW'
    _, CSKnmis, CSKaris, _ = utils.COPKmeansClusteringDBA(data, labels, n_clusters, ML, CL, max_iter=args.max_iter, trial=args.trial, metric=args.metric, initialization=args.initialization, type_=args.type, verbos=False)
    
    outdir = f'{args.Outdir}'
    if os.path.exists(f'{outdir}/{args.DataName}_{args.DataType}_{args.fr}_{args.nfr}_Score_{args.trialNumber}.csv' ):
        exit()
    with open(f'{outdir}/{args.DataName}_{args.DataType}_{args.fr}_{args.nfr}_Score_{args.trialNumber}.csv' , 'w') as f:
        f.write('Dataset,Data,Scope,Algorithm,Metric,fr,NMI,NMI_std,ARI,ARI_std')
        if args.Alldata:        
            f.write(f'\n{args.DataName},{args.DataType},All,COPKmean,{metric},{args.fr},{np.mean(CSKnmis)},{np.std(CSKnmis)},{np.mean(CSKaris)},{np.std(CSKaris)}')
        else:
            f.write(f'\n{args.DataName},{args.DataType},Train,COPKmean,{metric},{args.fr},{np.mean(CSKnmis)},{np.std(CSKnmis)},{np.mean(CSKaris)},{np.std(CSKaris)}')
