import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import CDPS_model as CDPS
import torch
import time
import gc
from glob import glob
import GPUtil
from GPUtil import showUtilization as gpu_usage
import pynvml
import pandas as pd
import sys
sys.path.append('../bin/')
import Utilities as utils

def cuda_availableGPU( threshold=50,cudaID=None):
    cmspgID = cuda_memorySummaryPerGPUID()
    maxID = max(cmspgID, key=cmspgID.get,default=0) 
    if cudaID == None:
        return maxID if cmspgID[maxID] >= threshold else None
    else:
        return cudaID if cmspgID[cudaID] >= threshold else None
    
def cuda_memorySummaryPerGPUID():
    ''' saves the free memory in the gpus identified by there id
        {gpuid: free_memory}
    '''
    cudanum = range(torch.cuda.device_count())
    cudamempynvml = {}
    pynvml.nvmlInit()
    for i in cudanum:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        cudamempynvml[i] = (mem_info.free/mem_info.total)*100
        print(f'Memory info {i}:memory_free; {(mem_info.free/mem_info.total):0.2f}')
    return cudamempynvml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Model_Fit',description='Fit the model with respect to different conditions', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('Outdir', help="Absolute path to a directory to write the outputs to.\n"
                        "Final and checkpoints Models are saved to <Outdir>/<DataName>/Model.\n"
                        "The intialization model is saved to <Outdir>/<DataName>/InitialzationModel.\n"
                        "The embeding are saved to <Outdir>/<DataName>/Embedings.\nResults  are saved to <Outdir>/Results.", type=str)
    parser.add_argument('DataName', help='Dataset Name', type=str)
    parser.add_argument('DataPath', help='Dataset DIR path', type=str)
    parser.add_argument('--CheckpointModel', nargs="?", dest='CheckpointModel', help='Checkpoint Model to continue from', type=str, default="auto")
    parser.add_argument('--LoadInitShapeletsBeta', nargs="?", dest='LoadInitShapeletsBeta', help='Load the initialized Shapelets and Beta wegihts', type=str)
    parser.add_argument('--InitShapeletsBeta', dest='InitShapeletsBeta', help='Initialize the Shapelets and Beta weights', default=False, action='store_true')
    parser.add_argument("--fr", nargs="?", dest='fr', default=None, type=float, help='Constraint fraction from dataset')
    parser.add_argument("--DTWmax", nargs="?", dest='DTWmax', default=None, type=float, help='Maximum DTW similarity ')
    parser.add_argument("--gamma", nargs="?", dest='gamma', default=None, type=float, help='CL Learning Force')
    parser.add_argument("--alpha", nargs="?", dest='alpha', default=None, type=float, help='ML Learning Force')
    parser.add_argument("--Constraints", dest='Constraints', help="Constraints path dir",nargs="?", default=None, type=str)
    parser.add_argument("--shapelet_max_scale", dest='shapelet_max_scale', help='Maximum shapelet scale; yields maximum shapelet length', nargs="?", default=3, type=int)
    parser.add_argument("--min_shapelet_length", dest='min_shapelet_length', help='Minimum length of a shapelet_length', nargs="?", default=0.15, type=float)
    parser.add_argument("--ratio_n_shapelets", dest='ratio_n_shapelets', help='Order(ratio) of shapelets number', nargs="?", default=10, type=int)
    parser.add_argument('--ple', dest="ple", help='print loss every:', type=int,default=50)
    parser.add_argument("--nkiter", nargs="?", dest='nkiter', default=500, type=int, help='Number of Epoch')
    parser.add_argument("--checkptsaveEach", nargs="?", dest='checkptsaveEach', default=None, type=int, help='To save a checkpoint after a constant number of Epochs')
    parser.add_argument("--bsize", help="Batch size", dest='bsize',nargs="?", default=30, type=int)
    parser.add_argument("--bCsize", help="Number of Constiraint pairs in Batch ", dest='bCsize',nargs="?", default=None, type=int)
    parser.add_argument("--learning_rate", help="Convergence rate ", dest='learning_rate',nargs="?", default=0.01, type=float)
    parser.add_argument('--cuda_ID', dest='cuda_ID', help='GPU ID', default=None, type=int)
    parser.add_argument('--nfr', help='Number of given constraint ratio set', type=int, default=None)
    parser.add_argument('--period', help='Period to reset the gamma alpha value', type=int, default=None)
    parser.add_argument("--Savescore", help="Save ARI and NMI scors", dest='Savescore', default=False, action='store_true')
    parser.add_argument('--disable_cuda', dest='disable_cuda', help='Disable CUDA', default=False, action='store_true')
    parser.add_argument('--Alldata', dest='Alldata', help='Use test and train for training', default=False, action='store_true')
    parser.add_argument('--zNormalize', dest='zNormalize', help='Z-Normalize the dataset before training', default=False, action='store_true')
    # TO DO
    parser.add_argument('--citer', dest='citer', help='Schedualing (deacy) for alpha and gamma', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    
    # Creating the output directory where model, result, and embedings are saved    
    if not args.InitShapeletsBeta:
        if args.Constraints != None:
            Outdir = f'{args.Outdir}/{args.Imp}/{args.DataName}/alpha_{args.alpha}_gamma_{args.gamma}_fr_{args.fr:.2f}_nfr_{args.nfr}/lmin_{args.min_shapelet_length}_shapelet_max_scale_{args.shapelet_max_scale}_ratio_n_shapelets_{args.ratio_n_shapelets}'
            if not os.path.exists(f'{Outdir}'):
                    os.makedirs(Outdir)
            ListofModels = glob(f"{Outdir}/Model/*.model") # Will never catch the init*.model so no nesed to remove it
            print(f'Checking if Final model exist in:\n{Outdir}')
            for f in ListofModels:
                if "Final" in f:
                    print("There is already a finished trained model, please change the directoy")
                    exit()
        else:
            Outdir = f'{args.Outdir}/{args.Imp}/{args.DataName}/Noconstraints/lmin_{args.min_shapelet_length}_shapelet_max_scale_{args.shapelet_max_scale}_ratio_n_shapelets_{args.ratio_n_shapelets}'
            if not os.path.exists(f'{Outdir}'):
                os.makedirs(Outdir)
            ListofModels = glob(f"{Outdir}/Model/*.model")
            print(f'Checking if Final model exist in:\n{Outdir}')
            for f in ListofModels:
                if "Final" in f:
                    print("There is already a finished trained model, please change the directoy")
                    exit()

    #Check if using cuda or not
    print('Checking if cuda availables')
    if not args.disable_cuda and torch.cuda.is_available():
        print("using cuda")
        while cuda_availableGPU(threshold=40,cudaID=args.cuda_ID) == None:
            print(f'Waiting for available GPU with threshold 40 percentage of free')
            time.sleep(60)
        cuda_ID = cuda_availableGPU(threshold=40,cudaID=args.cuda_ID)
        device = torch.device(f'cuda:{cuda_ID}')
        print(f"Using cuda device {device}")

    else:
        device = torch.device(f'cpu')
        print(f"Using device {device}")
    
    #Loading data
    if args.DataName != 'Sud':    
        Trdata, Trgt_labels = utils.load_dataset_ts(args.DataName,'TRAIN',args.DataPath)
        Tsdata, Tsgt_labels = utils.load_dataset_ts(args.DataName,'TEST',args.DataPath)
    elif args.DataName == 'Sud':
        Trdata = np.loadtxt(f'{args.DataPath}/train/Sud.data')
        Trdata = Trdata.reshape(Trdata.shape[0],Trdata.shape[1],1)
        Trgt_labels = np.loadtxt(f'{args.DataPath}/train/Sud.labels')
        Tsdata = np.loadtxt(f'{args.DataPath}/test/Sud.data')
        Tsdata = Tsdata.reshape(Tsdata.shape[0],Tsdata.shape[1],1)
        Tsgt_labels = np.loadtxt(f'{args.DataPath}/test/Sud.labels')
    if args.Alldata:
        print("Using both train and test set")
        data = np.concatenate((Trdata,Tsdata))
        labels = np.concatenate((Trgt_labels,Tsgt_labels))
        n_clusters = utils.n_classes(labels)
        if args.zNormalize:
            data = (data - data.mean())/(data.std())
        del Trdata, Trgt_labels, Tsdata, Tsgt_labels
        gc.collect()
        alldata = "Alldata"
    else:
        print("Using train set")
        data = Trdata
        labels = Trgt_labels
        n_clusters = utils.n_classes(labels)
        if args.zNormalize:
            data = (data - data.mean())/(data.std())
            Tsdata = (Tsdata - Tsdata.mean())/(Tsdata.std())
        del Trdata, Trgt_labels
        gc.collect()
        alldata = "Traindata"
   
    #Initializing shapelets    
    if args.InitShapeletsBeta:
            print("Initializing the model wieghts; shapelets and beta")
            lmin = args.min_shapelet_length
            s = args.shapelet_max_scale
            rs = [lmin*i for i in range(1,s+1)]
            shapelet_lengths = {}
            for sz in [int(p * data.shape[1]) for p in rs]:
                n_shapelets = int(np.log(data.shape[1] - sz) * args.ratio_n_shapelets)  
                shapelet_lengths[sz] = n_shapelets
            print(f'Shapelet blocks {shapelet_lengths}')
            print('Calculating DTW max')
            DTWmax = utils.DTWMax_dtwPython(data)
            model = CDPS.CDPSModel(n_shapelets_per_size=shapelet_lengths,ts_dim=data.shape[2],lr=args.learning_rate, 
                                     epochs=args.nkiter, batch_size=args.bsize, DTWmax=DTWmax, device=device, verbose=True, 
                                     saveloss=args.Savescore, citer=args.citer, ple=args.ple)
            print('Initializing')
            model._init_params(X=data)
            Outdir = f'{args.Outdir}/{args.Imp}/{args.DataName}/InitialzationModel'
            if not os.path.exists(Outdir):
                    os.makedirs(Outdir)
            model.model_save(f'{Outdir}/Initialization_model_{args.DataName}_cuda_{not args.disable_cuda}_Alldata_{args.Alldata}.model')   
            print("Finished Initializing...\nExiting...")
            exit()
   
    #Loading Constraints
    if args.Constraints != None:
        if args.DataName != 'Sud':
            ML = np.load(os.path.join(args.Constraints,args.DataName+ f'_{alldata}/{args.DataName}_{alldata}_fr_{args.fr:.2f}_ML_tn_{args.nfr}.npy'))
            CL = np.load(os.path.join(args.Constraints,args.DataName + f'_{alldata}/{args.DataName}_{alldata}_fr_{args.fr:.2f}_CL_tn_{args.nfr}.npy'))
        elif args.DataName == 'Sud':
            constraints = pd.read_csv(f'{args.Constraints}/train/Sud_{args.fr}_{args.nfr}.constraints', names=['idx1', 'idx2', 'con'],sep='\t')
            ML  = constraints[constraints.con == 1][['idx1', 'idx2']].values - 1
            CL  = constraints[constraints.con == -1][['idx1', 'idx2']].values - 1
    else:
            ML ,CL = None, None
   
    #Loading/building the model        
    if not args.LoadInitShapeletsBeta:
        print("Initializing the model.")
        lmin = args.min_shapelet_length
        s = args.shapelet_max_scale
        rs = [lmin*i for i in range(1,s+1)]
        r = 10
        shapelet_lengths = {}
        for sz in [int(p * data.shape[1]) for p in rs]:
            n_shapelets = int(np.log(data.shape[1] - sz) * args.ratio_n_shapelets)  
            shapelet_lengths[sz] = n_shapelets
        DTWmax = utils.DTWMax_dtwPython(data)# change
        model = CDPS.CDPSModel(n_shapelets_per_size=shapelet_lengths,ts_dim=data.shape[2],lr=args.learning_rate, 
                                 epochs=args.nkiter, fr=args.fr, batch_size=args.bsize, DTWmax=DTWmax, ML=ML, CL=CL,
                                 gamma=args.gamma, period=args.period, alpha=args.alpha, constraints_in_batch=args.bCsize,
                                 device=device, verbose=True, saveloss=args.Savescore, citer=args.citer, ple=args.ple)
        model._init_params(X=data) # Init beta and shapelets
        lastepoch = 0
    elif ListofModels != [] and args.CheckpointModel:
        if args.CheckpointModel == 'auto':
            print("Loading CheckpointModel")
            lastepoch = max([int(os.path.basename(f).split('_')[2]) for f in ListofModels]) 
            lastepoch = 0 if lastepoch == [] else lastepoch
            print(f'Continue training from epoch {lastepoch}')
            for f in ListofModels:
                if str(lastepoch) in f:
                    lastchekpointmodel  = f
                    break
            model =  CDPS.CDPSModel.model_load(lastchekpointmodel,device=device)
            model.device,model.loss_.device=device,device
            model.loss_.saveloss=args.Savescore

        else:
            print(f"Loading checkpoint:\n{args.CheckpointModel}")
            model =  CDPS.CDPSModel.model_load(args.CheckpointModel,device=device)
            model.device,model.loss_.device=device,device
            model.loss_.saveloss=args.Savescore
    else:
        print("Loading Initailized shapelets and beta. Initializing model parameters.")
        initmodel = f'{args.LoadInitShapeletsBeta}/Initialization_model_{args.DataName}_cuda_{not args.disable_cuda}_Alldata_{args.Alldata}.model'
        model = CDPS.CDPSModel.model_load(initmodel,device=device)
        model.ML  = ML
        model.CL = CL
        model.fr, model.loss_.fr = args.fr, args.fr
        model.gamma, model.loss_.gamma = args.gamma, args.gamma
        model.alpha, model.loss_.alpha = args.alpha, args.alpha 
        model.constraints_in_batch = args.bCsize if args.bCsize else  model.constraints_in_batch
        model.ple = args.ple
        model.device,model.loss_.device=device,device
        model.loss_.saveloss=args.Savescore
        lastepoch = 0

    #Training
    if not os.path.exists(f'{Outdir}/Model'):
            os.makedirs(f'{Outdir}/Model')

    print("Training")
    if args.checkptsaveEach == None:
        bftime = time.time()
        model.fit(data, init_=False)
        Trtime= time.time() - bftime
    else:
        model.epochs = args.checkptsaveEach
        model.ple=args.ple
        gpEpochleft = args.nkiter - lastepoch
        Trtime = 'Not_recorded'
        epg = gpEpochleft//args.checkptsaveEach
        for e in range(epg):
            model.fit(data,init_=False)
            print(f"Finshed epoch group {e}/{epg}")
            if (e+1) != gpEpochleft//args.checkptsaveEach:
                print(f"Saveing checkpoint: {Outdir}/Model/Checkpoint_Epoch_{(e+1+lastepoch)*args.checkptsaveEach}_model_cuda_{not args.disable_cuda}_Alldata_{args.Alldata}.model")
                model.model_save(f'{Outdir}/Model/Checkpoint_Epoch_{(e+1+lastepoch)*args.checkptsaveEach}_model_cuda_{not args.disable_cuda}_Alldata_{args.Alldata}.model')
    
    
    # Saving the Final model
    print(f'Saving Final model: {Outdir}/Model/Final_model_cuda_{not args.disable_cuda}_Alldata_{args.Alldata}.model')
    model.model_save(f'{Outdir}/Model/Final_model_cuda_{not args.disable_cuda}_Alldata_{args.Alldata}.model')   
    
    # Saving the Data Embedings
    if not os.path.exists(f'{Outdir}/Embedings'):
        os.makedirs(f'{Outdir}/Embedings', exist_ok=True)
    if args.Alldata:
        print(f'Saving Data Embeding: Alldata')
        embTslearnNUmpy = []
        for row in data:
            embTslearnNUmpy.append(model._features(CDPS.tslearn2torch(row.reshape(1,-1,data.shape[2]),device)).detach().cpu().numpy())
        embTslearnNUmpy = np.concatenate(embTslearnNUmpy)
        #embTslearn = model._features(CLDPS.tslearn2torch(data,device))
        #embTslearnNUmpy = embTslearn.cpu().detach().numpy()
        np.save(f'{Outdir}/Embedings/TransformedData_{not args.disable_cuda}_Alldata_{args.Alldata}', embTslearnNUmpy)
    else:
        print(f'Saving Data Embeding: Train')
        embTslearnNUmpy = []
        for row in data:
            if args.Imp =='UNI':
                embTslearnNUmpy.append(model._features(CDPS.tslearn2torch(row.reshape(1,-1,data.shape[2]),device)).detach().cpu().numpy())
        embTslearnNUmpy = np.concatenate(embTslearnNUmpy)
#        embTslearn = model._features(CLDPS.tslearn2torch(data,device))
#        embTslearnNUmpy = embTslearn.cpu().detach().numpy()
        np.save(f'{Outdir}/Embedings/TransformedData_{not args.disable_cuda}_Train', embTslearnNUmpy)
        print(f'Saving Data Embeding: Test')
        embTslearnNUmpytst = []
        for row in Tsdata:
            embTslearnNUmpytst.append(model._features(CDPS.tslearn2torch(row.reshape(1,-1,data.shape[2]),device)).detach().cpu().numpy())
        embTslearnNUmpytst = np.concatenate(embTslearnNUmpytst)
#        embTslearntst = model._features(CLDPS.tslearn2torch(Tsdata,device))
#        embTslearnNUmpytst = embTslearn.cpu().detach().numpy()
        np.save(f'{Outdir}/Embedings/TransformedData_{not args.disable_cuda}_Test', embTslearnNUmpytst)
   
    #Saving the training loss and clustering test.        
    if args.Savescore:
        print("Doing the tests and saving the results in Results")
        Resultdir = f'{Outdir}/Results'
        if not os.path.exists(Resultdir):
            os.makedirs(Resultdir, exist_ok=True)
        print(f'Saving Figures:')
        if args.Constraints != None:
            plt.figure();plt.title('ML Loss');plt.plot([i for i in range(len(model.loss_.lossMLtrack))],model.loss_.lossMLtrack);
            plt.savefig("%s/%s_%s.svg" % (Resultdir,args.DataName, 'ML_LOSS'),  format='svg', dpi=1200);plt.close()
            plt.figure();plt.title('CL Loss');plt.plot([i for i in range(len(model.loss_.lossCLtrack))],model.loss_.lossCLtrack);
            plt.savefig("%s/%s_%s.svg" % (Resultdir,args.DataName, 'CL_LOSS'),  format='svg', dpi=1200);plt.close()
        plt.figure();plt.title('Loss per epoch');plt.plot([i for i in range(len(model.losstrack))],model.losstrack);
        plt.savefig("%s/%s_%s.svg" % (Resultdir,args.DataName, 'Loss_per_EPOCH'),  format='svg', dpi=1200);plt.close()
        print(f'performing Clsutering on {args.Alldata}')
        if args.Alldata:        
            # _, CSKnmis, CSKaris = utils.KMeanClustering(embTslearnNUmpy, n_clusters, labels)
            _, CSKnmis, CSKaris = utils.COPKmeansClusteringDBA(embTslearnNUmpy, labels, n_clusters, [], [], initialization='random',
                                   max_iter=100, trial=10,metric='l2_distance', type_='dependent',verbos=False)

        else:
            # _, CSKnmis, CSKaris = utils.KMeanClustering(embTslearnNUmpy, n_clusters, labels)
            # _, CSKnmistst, CSKaristst = utils.KMeanClustering(embTslearnNUmpytst, n_clusters, Tsgt_labels)
            
            _, CSKnmis, CSKaris, _ = utils.COPKmeansClusteringDBA(embTslearnNUmpy.reshape(embTslearnNUmpy.shape[0],embTslearnNUmpy.shape[1],1), labels, n_clusters, [], [], initialization='random',
                                   max_iter=100, trial=10, metric='l2_distance', type_='dependent',verbos=False)
            _, CSKnmistst, CSKaristst, _ = utils.COPKmeansClusteringDBA(embTslearnNUmpytst.reshape(embTslearnNUmpytst.shape[0],embTslearnNUmpytst.shape[1],1), Tsgt_labels, n_clusters, [], [], initialization='random',
                                   max_iter=100, trial=10,metric='l2_distance', type_='dependent',verbos=False)
            temp = np.concatenate([embTslearnNUmpy,embTslearnNUmpytst])
            _, CSKnmiststall, CSKariststall, _ = utils.COPKmeansClusteringDBA(temp.reshape(temp.shape[0],temp.shape[1],1), np.concatenate(labels, Tsgt_labels), n_clusters, [], [], initialization='random',
                                   max_iter=100, trial=10,metric='l2_distance', type_='dependent',verbos=False)

        print(f'Saving clustering Results: {args.DataName}_Score.csv')    
        with open(f'{Resultdir}/{args.DataName}_Score.csv' , 'a') as f:
            f.write('Data,Scope,Algorithm,Metric,Gamma,Alpha,fr,minSlen,ratioS,SmaxNum,TrainingTime,NMI,NMI_std,ARI,ARI_std')
            if args.Alldata:        
                f.write(f'\nEmbeding,All,Kmean,Euclidean,{model.gamma},{model.alpha},{model.fr},{args.min_shapelet_length},{args.ratio_n_shapelets},{args.shapelet_max_scale},{Trtime},{np.mean(CSKnmis)},{np.std(CSKnmis)},{np.mean(CSKaris)},{np.std(CSKaris)}')
            else:
                f.write(f'\nEmbeding,Train,Kmean,Euclidean,{model.gamma},{model.alpha},{model.fr},{args.min_shapelet_length},{args.ratio_n_shapelets},{args.shapelet_max_scale},{Trtime},{np.mean(CSKnmis)},{np.std(CSKnmis)},{np.mean(CSKaris)},{np.std(CSKaris)}')
                f.write(f'\nEmbeding,Test,Kmean,Euclidean,{model.gamma},{model.alpha},{model.fr},{args.min_shapelet_length},{args.ratio_n_shapelets},{args.shapelet_max_scale},{Trtime},{np.mean(CSKnmistst)},{np.std(CSKnmistst)},{np.mean(CSKaristst)},{np.std(CSKaristst)}')
                f.write(f'\nEmbeding,all,Kmean,Euclidean,{model.gamma},{model.alpha},{model.fr},{args.min_shapelet_length},{args.ratio_n_shapelets},{args.shapelet_max_scale},{Trtime},{np.mean(CSKnmistst)},{np.std(CSKnmistst)},{np.mean(CSKaristst)},{np.std(CSKaristst)}')
 
    gc.collect()        
    print('Freed allocated memory...')
    print('Finished...\nExiting...')
    

            


