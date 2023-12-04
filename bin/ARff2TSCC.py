#%%
from scipy.io import arff
import numpy as np
import pandas as pd
import os
import Utilities as utils
from scipy.stats import zscore
#%%
def UniarffTotxt(dataset, datasetpathdir, types, output, metric='dtw'):
    if types == 'alldata' or types == 'train':
        dataTrain, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff")
        dataTest, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff")
        data = np.hstack([dataTrain,dataTest])
    elif types == 'test':
        data, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff")
    elif types == 'trainonly':
        data, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff")
        
    dfData = pd.DataFrame(data)
    data = dfData.iloc[:,:-1].to_numpy()
    outputdir = f'{output}/{dataset}/{types}'
    print(f'outputdir {outputdir}')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    labels = dfData.iloc[:,-1].to_numpy().astype('<U8')
    labels = utils.labelsToint(labels)
    print(f'Saving labels: {dataset}.labels')
    np.savetxt(f'{outputdir}/{dataset}.labels',np.array([labels]).T,fmt='%i')
    k = np.unique(labels).shape[0]
    print(f'Saving number of classes: {dataset}.k')
    np.savetxt(f'{outputdir}/{dataset}.k',np.array([k]).T,fmt='%i')
    nfeatures =  1
    print(f'Saving number of features: {dataset}.nfeatures')
    np.savetxt(f'{outputdir}/{dataset}.nfeatures',np.array([nfeatures]).T,fmt='%i')
    np.savetxt(f'{outputdir}/{dataset}.metric',np.array([metric]).T,fmt='%s')
    if os.path.exists(f'{outputdir}/{dataset}.data'):
        print(f'file exists removing {outputdir}/{dataset}.data')
        os.remove(f'{outputdir}/{dataset}.data')
    print(f'Saving data: {dataset}.data')
    with open(f'{outputdir}/{dataset}.data','a') as outdata:
        print(f'Generating .data file: {outputdir}/{dataset}.data ')
        for i in range(data.shape[0]):
            timeseries = data[i].view(float).reshape((-1,data[i].shape[0]))
            timeseries = zscore(timeseries,axis=1)
            np.savetxt(outdata,timeseries.T.reshape(1,-1),delimiter='\t',newline='\n')
#%%  
def MultiarffTotxt(dataset, datasetpathdir, types, output, metric='dtw'):
    if types == 'alldata' or types == 'train':
        dataTrain, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff")
        dataTest, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff")
        data = np.hstack([dataTrain,dataTest])
    elif types == 'test':
        data, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff")
    elif types == 'trainonly':
        data, _ = arff.loadarff(f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff")
        
    dfData = pd.DataFrame(data)
    data = dfData.iloc[:,0]
    outputdir = f'{output}/{dataset}/{types}'
    print(f'outputdir {outputdir}')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    labels = dfData.iloc[:,1].to_numpy().astype('<U8')#.astype(float).astype(int)
    labels = utils.labelsToint(labels)
    print(f'Saving labels: {dataset}.labels')
    np.savetxt(f'{outputdir}/{dataset}.labels',np.array([labels]).T,fmt='%i')
    k = np.unique(labels).shape[0]
    print(f'Saving number of classes: {dataset}.k')
    np.savetxt(f'{outputdir}/{dataset}.k',np.array([k]).T,fmt='%i')
    nfeatures =  data[0].shape[0]
    print(f'Saving number of features: {dataset}.nfeatures')
    np.savetxt(f'{outputdir}/{dataset}.nfeatures',np.array([nfeatures]).T,fmt='%i')
    np.savetxt(f'{outputdir}/{dataset}.metric',np.array([metric]).T,fmt='%s')
    if os.path.exists(f'{outputdir}/{dataset}.data'):
        print(f'file exists removing {outputdir}/{dataset}.data')
        os.remove(f'{outputdir}/{dataset}.data')
    print(f'Saving data: {dataset}.data')
    with open(f'{outputdir}/{dataset}.data','a') as outdata:
        print(f'Generating .data file: {outputdir}/{dataset}.data ')
        for i in range(data.shape[0]):
            timeseries = data[i].view(float).reshape(data[i].shape + (-1,))
            timeseries = zscore(timeseries,axis=1)
            np.savetxt(outdata, timeseries.T.reshape(1,-1), delimiter='\t', newline='\n')
    
#%%

if __name__ == main:
    
    parser = argparse.ArgumentParser(prog='ARFF2TSCC',description='Transforming ARFF representation to TSCC.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('output', help="Absolute path to a directory to write the outputs to.", type=str)
    parser.add_argument('dataset', help='Dataset Name', type=str)
    parser.add_argument('datasetpathdir', help='Dataset DIR path', type=str)
    parser.add_argument('--DataType', help='Multivariate or Univariate Series.', type=str, default='Univariate')
    parser.add_argument('--split_type', help='Split type of dataset. Either trainonly, test and alldata.', type=str, default='Univariate')
    parser.add_argument('--metric', help='type of metric to use in TSCC code. Either dtw or euclidean', type=str, default='dtw')
    args = parser.parse_args()
    print(args)    
    
    print(f'Transforming Dataset {dataset} to TSCC format:')
    if DataType == 'Multivariate':
        MultiarffTotxt(dataset, datasetpathdir, split_type, output=output, metric=metric)
    elif  DataType == 'Univariate':
        UniarffTotxt(dataset, datasetpathdir, split_type, output=output, metric=metric)


