import Utilities as utils
import numpy as np
import argparse
import gc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Generate Constraints',description='Gerenate ML/CL Constraints, each files containts the pairwise index of constrainted points')
    parser.add_argument('Outdir', help='DIR path to save the Models in <MOutpath>/Models/', type=str)
    parser.add_argument('dataName', help='Dataset Name', type=str)
    parser.add_argument('dataPath', help='Dataset DIR path', type=str)
    parser.add_argument('--fr', action='append', help='Constraint fraction from dataset', type=str, required=True)
    parser.add_argument('--Alldata', dest='Alldata', help='Use test and train for training', default=False, action='store_true')
    parser.add_argument('--nfr', help='Number of given constraint ratio set', type=int, default=None) 
    args = parser.parse_args()
    print(args)

        
    Trdata, Trgt_labels = utils.load_dataset_ts(args.dataName,'TRAIN',args.dataPath)
    Tsdata, Tsgt_labels = utils.load_dataset_ts(args.dataName,'TEST',args.dataPath)
    if args.Alldata:
        print("Training on train and test set")
        data = np.concatenate((Trdata,Tsdata))
        labels = np.concatenate((Trgt_labels,Tsgt_labels))
        labels = utils.labelsToint(labels)
        n_clusters = utils.n_classes(labels)
        del Trdata, Trgt_labels, Tsdata, Tsgt_labels
        gc.collect()
        NAME = args.dataName+'_Alldata'
    else:
        print("Training on the train set")
        data = Trdata
        labels = Trgt_labels
        labels = utils.labelsToint(labels)
        del Trdata, Trgt_labels
        gc.collect()
        NAME = args.dataName+'_Traindata'
    utils.save_constraints(data, labels, np.array(args.fr).astype(np.float32), args.Outdir, NAME, nb=args.nfr)
    