import os
from glob import glob
import numpy as np
import Utilities  as utils
import sys
sys.path.append('../CDPS/')
import CDPS_model as CDPS

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='ARFF2TSCC',description='Transforming ARFF representation to TSCC.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('output', help="Absolute path to a directory to write the outputs to.", type=str)
    parser.add_argument('outname', help="Name  of the output file.", type=str)
    parser.add_argument('dataset', help='Dataset Name', type=str)
    parser.add_argument('DataPath', help='Dataset DIR path', type=str)
    parser.add_argument('dirs', help='Absolute path to the directory containing the model checkpoints and the last model.', type=str)
    parser.add_argument('--DataType', help='Multivariate or Univariate Series.', type=str, default='Univariate')
    parser.add_argument('--device', help='specify if the model is on "cpu" or "gpu".', type=str, default='cpu')
    parser.add_argument('--Alldata', dest='Alldata', help='Use test and train for training', default=False, action='store_true')
    parser.add_argument('--disable_cuda', dest='disable_cuda', help='Disable CUDA', default=False, action='store_true')
    args = parser.parse_args()
    print(args)    
    files = glob(args.dirs)

    for CheckpointModel in files:
        if f'Alldata_{args.Alldata}' in CheckpointModel:
            type_ = CheckpointModel.split('/')[7]
            type_ = type_ if 'alpha' not in type_ else 'Constraints'
    
            dataset = CheckpointModel.split('/')[6]
            fracIter = CheckpointModel.split('/')[7]
            Outdir = f'{args.output}/{args.outname}/{args.dataset}/{fracIter}/Embedings/'
            if not os.path.exists(Outdir):
                os.makedirs(Outdir)
            print(Outdir)
            #Loading data    
            Trdata, Trgt_labels = utils.load_dataset_ts(args.dataset,'TRAIN',args.DataPath)
            Tsdata, Tsgt_labels = utils.load_dataset_ts(args.dataset,'TEST',args.DataPath)
            if args.Alldata:
                print("Using both train and test set")
                data = np.concatenate((Trdata,Tsdata))
                labels = np.concatenate((Trgt_labels,Tsgt_labels))
                n_clusters = utils.n_classes(labels)
                del Trdata, Trgt_labels, Tsdata, Tsgt_labels
                alldata = "Alldata"
            else:
                print("Using train set")
                data = Trdata
                labels = Trgt_labels
                n_clusters = utils.n_classes(labels)
                del Trdata, Trgt_labels
                alldata = "Traindata"
            model = CDPS.CDPSModel.model_load(CheckpointModel,device='cpu')
            if not os.path.exists(f'{Outdir}/Embedings'):
                os.makedirs(f'{Outdir}/Embedings')
            if args.Alldata:
                print(f'Saving Data Embeding: Alldata')
                embTslearnNUmpy = []
                for row in data:
                    embTslearnNUmpy.append(model._features(CDPS.tslearn2torch(
                        row.reshape(1, -1, data.shape[2]), args.device)).detach().cpu().numpy())
                embTslearnNUmpy = np.concatenate(embTslearnNUmpy)
                np.save(f'{Outdir}/Embedings/TransformedData_{not args.disable_cuda}_Alldata_{args.Alldata}', embTslearnNUmpy)
            else:
                print(f'Saving Data Embeding: Train')
                embTslearnNUmpy = []
                for row in data:
                    embTslearnNUmpy.append(model._features(CDPS.tslearn2torch(row.reshape(1,-1,data.shape[2]),args.device)).detach().cpu().numpy())
                embTslearnNUmpy = np.concatenate(embTslearnNUmpy)
                np.save(f'{Outdir}/Embedings/TransformedData_{not args.disable_cuda}_Train', embTslearnNUmpy)
                print(f'Saving Data Embeding: Test')
                embTslearnNUmpytst = []
                for row in Tsdata:
                    embTslearnNUmpytst.append(model._features(CDPS.tslearn2torch(row.reshape(1,-1,data.shape[2]),args.device)).detach().cpu().numpy())
                embTslearnNUmpytst = np.concatenate(embTslearnNUmpytst)
                np.save(f'{Outdir}/Embedings/TransformedData_{not args.disable_cuda}_Test', embTslearnNUmpytst)
         
