"""
ARff2TSCC.py
Transfrom .ARFF extension tp .TSCC extension.
"""
import os
import argparse
import numpy as np
import pandas as pd
import Utilities as utils
from scipy.io import arff
from scipy.stats import zscore


def UniarffTotxt(dataset, datasetpathdir, types, output, metric='dtw'):
    '''
    For Univeraite time series.
    '''

    if types == 'alldata' or types == 'train':
        train_arff = f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff"
        datatrain, _ = arff.loadarff(train_arff)
        test_arff = f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff"
        datatest, _ = arff.loadarff(test_arff)
        data = np.hstack([datatrain, datatest])
    elif types == 'test':
        test_arff = f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff"
        data, _ = arff.loadarff(test_arff)
    elif types == 'trainonly':
        train_arff = f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff"
        data, _ = arff.loadarff(train_arff)

    dfdata = pd.DataFrame(data)
    data = dfdata.iloc[:, : -1].to_numpy()
    outputdir = f'{output}/{dataset}/{types}'
    print(f'outputdir {outputdir}')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    labels = dfdata.iloc[:, -1].to_numpy().astype('<U8')
    labels = utils.labelsToint(labels)
    print(f'Saving labels: {dataset}.labels')
    np.savetxt(f'{outputdir}/{dataset}.labels',
               np.array([labels]).T,
               fmt='%i')
    k = np.unique(labels).shape[0]
    print(f'Saving number of classes: {dataset}.k')
    np.savetxt(f'{outputdir}/{dataset}.k',
               np.array([k]).T,
               fmt='%i')
    nfeatures = 1
    print(f'Saving number of features: {dataset}.nfeatures')
    np.savetxt(f'{outputdir}/{dataset}.nfeatures',
               np.array([nfeatures]).T,
               fmt='%i')
    np.savetxt(f'{outputdir}/{dataset}.metric',
               np.array([metric]).T,
               fmt='%s')
    if os.path.exists(f'{outputdir}/{dataset}.data'):
        print(f'file exists removing {outputdir}/{dataset}.data')
        os.remove(f'{outputdir}/{dataset}.data')
    print(f'Saving data: {dataset}.data')
    with open(f'{outputdir}/{dataset}.data', 'a') as outdata:
        print(f'Generating .data file: {outputdir}/{dataset}.data ')
        for i in range(data.shape[0]):
            timeseries = data[i].view(float).reshape((-1, data[i].shape[0]))
            timeseries = zscore(timeseries, axis=1)
            np.savetxt(outdata,
                       timeseries.T.reshape(1, -1),
                       delimiter='\t',
                       newline='\n')


def MultiarffTotxt(dataset, datasetpathdir, types, output, metric='dtw'):
    '''
    For multivaraite time series.
    '''

    if types == 'alldata' or types == 'train':
        train_arff = f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff"
        datatrain, _ = arff.loadarff(train_arff)
        test_arff = f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff"
        datatest, _ = arff.loadarff(test_arff)
        data = np.hstack([datatrain, datatest])
    elif types == 'test':
        test_arff = f"{datasetpathdir}/{dataset}/{dataset}_TEST.arff"
        data, _ = arff.loadarff(test_arff)
    elif types == 'trainonly':
        train_arff = f"{datasetpathdir}/{dataset}/{dataset}_TRAIN.arff"
        data, _ = arff.loadarff(train_arff)

    dfdata = pd.DataFrame(data)
    data = dfdata.iloc[:, 0]
    outputdir = f'{output}/{dataset}/{types}'
    print(f'outputdir {outputdir}')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    labels = dfdata.iloc[:, 1].to_numpy().astype('<U8')
    labels = utils.labelsToint(labels)
    print(f'Saving labels: {dataset}.labels')
    np.savetxt(f'{outputdir}/{dataset}.labels',
               np.array([labels]).T,
               fmt='%i')
    k = np.unique(labels).shape[0]
    print(f'Saving number of classes: {dataset}.k')
    np.savetxt(f'{outputdir}/{dataset}.k',
               np.array([k]).T,
               fmt='%i')
    nfeatures = data[0].shape[0]
    print(f'Saving number of features: {dataset}.nfeatures')
    np.savetxt(f'{outputdir}/{dataset}.nfeatures',
               np.array([nfeatures]).T,
               fmt='%i')
    np.savetxt(f'{outputdir}/{dataset}.metric',
               np.array([metric]).T,
               fmt='%s')
    if os.path.exists(f'{outputdir}/{dataset}.data'):
        print(f'file exists removing {outputdir}/{dataset}.data')
        os.remove(f'{outputdir}/{dataset}.data')
    print(f'Saving data: {dataset}.data')
    with open(f'{outputdir}/{dataset}.data', 'a') as outdata:
        print(f'Generating .data file: {outputdir}/{dataset}.data ')
        for i in range(data.shape[0]):
            timeseries = data[i].view(float).reshape(data[i].shape + (-1,))
            timeseries = zscore(timeseries, axis=1)
            np.savetxt(outdata,
                       timeseries.T.reshape(1, -1),
                       delimiter='\t',
                       newline='\n')


if __name__ == '__main__':

    TEXT = 'Transforming ARFF representation to TSCC.'
    foramtter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(prog='ARFF2TSCC',
                                     description=TEXT,
                                     formatter_class=foramtter)
    parser.add_argument('output',
                        help="Output dirctory absoloute path.",
                        type=str)
    parser.add_argument('dataset',
                        help='Dataset Name',
                        type=str)
    parser.add_argument('datasetpathdir',
                        help='Dataset DIR path',
                        type=str)
    parser.add_argument('--DataType',
                        help='Multivariate or Univariate Series.',
                        type=str,
                        default='Univariate')
    parser.add_argument('--split_type',
                        help='''Split dataset between train and test sets,
                        or not.''',
                        type=str,
                        default='alldata')
    parser.add_argument('--metric',
                        help='Use dtw or euclidean as a distance measure.',
                        type=str,
                        default='dtw')
    args = parser.parse_args()
    print(args)

    print(f'Transforming Dataset {args.dataset} to TSCC format:')
    if args.DataType == 'Multivariate':
        MultiarffTotxt(args.dataset,
                       args.datasetpathdir,
                       args.split_type,
                       output=args.output,
                       metric=args.metric)
    elif args.DataType == 'Univariate':
        UniarffTotxt(args.dataset,
                     args.datasetpathdir,
                     args.split_type,
                     output=args.output,
                     metric=args.metric)
