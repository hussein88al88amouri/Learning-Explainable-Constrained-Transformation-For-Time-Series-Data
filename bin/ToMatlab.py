#%%
import os
import numpy as np
import pandas as pd

#%%   -------------------------------------------------------------------------------------------------------------------------------------------------------
#     ---------------------------------------------To Matlab Constraints and Data----------------------------------------------------------------------------------
#     -------------------------------------------------------------------------------------------------------------------------------------------------------

def tabsaperateddata(dataset, emb, Outpath=None):
    OutEmb  = os.path.join(EmbedingsPath,f'{dataset}.txt') if not Outpath else  os.path.join(Outpath,f'{dataset}.data') 
    if not os.path.exists(Outpath):
        print(Outpath)
        os.makedirs(Outpath)
    np.savetxt(OutEmb, emb, delimiter='\t' ,newline='\n')
        
def tabserpatedconstraitns(dataset, ML, CL, fr, nfr, Outpath=None):
    if not os.path.exists(Outpath):
        print(Outpath)
        os.makedirs(Outpath)
    MLdf = pd.DataFrame(ML)
    MLdf = MLdf + 1
    MLdf[2] = 1
    CLdf = pd.DataFrame(CL)
    CLdf = CLdf + 1
    CLdf[2] = -1
    df = pd.concat([MLdf, CLdf])
    OutEmb  = os.path.join(EmbedingsPath,f'{dataset}.txt') if not Outpath else  os.path.join(Outpath,f'{dataset}_{fr}_{nfr}.constraints') 
    np.savetxt(OutEmb,df.to_numpy().astype(int), fmt='%i',delimiter='\t',newline='\n')

def getEmbeddingsSize(dataset, emb, Outpath=None):
    OutEmb  = os.path.join(EmbedingsPath,f'{dataset}.txt') if not Outpath else  os.path.join(Outpath,f'{dataset}.EmbFeatureSize') 
    if not os.path.exists(Outpath):
        print(Outpath)
        os.makedirs(Outpath)
    np.savetxt(OutEmb, np.array(emb.shape[2]), delimiter='\t' ,newline='\n')

    


    
datasets='ACSF1 ArrowHead BeetleFly Chinatown ChlorineConcentration CinCECGTorso Computers DistalPhalanxTW Earthquakes ECG5000 ECGFiveDays ElectricDevices EOGHorizontalSignal EOGVerticalSignal EthanolLevel FordA FordB FreezerRegularTrain FreezerSmallTrain GunPointMaleVersusFemale GunPointOldVersusYoung Ham HandOutlines Haptics HouseTwenty InlineSkate InsectEPGRegularTrain InsectEPGSmallTrain InsectWingbeatSound ItalyPowerDemand LargeKitchenAppliances MedicalImages MiddlePhalanxOutlineAgeGroup MiddlePhalanxOutlineCorrect MiddlePhalanxTW MixedShapesRegularTrain MixedShapesSmallTrain NonInvasiveFetalECGThorax1 OliveOil PhalangesOutlinesCorrect PigAirwayPressure PigArtPressure PigCVP ProximalPhalanxOutlineAgeGroup ProximalPhalanxOutlineCorrect ProximalPhalanxTW RefrigerationDevices SemgHandGenderCh2 SemgHandMovementCh2 ShapeletSim SmallKitchenAppliances SmoothSubspace SonyAIBORobotSurface1 SonyAIBORobotSurface2 Strawberry ToeSegmentation1 ToeSegmentation2 Trace TwoLeadECG TwoPatterns UMD UWaveGestureLibraryAll UWaveGestureLibraryX UWaveGestureLibraryY Wafer Wine WordSynonyms Worms WormsTwoClass Yoga'.split(' ')

embedingpathdirsFr03 = glob(f'{embedingpathdir}/**/alp*_fr_0.30_nfr_9/**/Em*/*')

for dataset in datasets:
    Outpath=f"/media/gancarsk/Samsung_T5/TestsForPaper/dataset/{dataset}/alldata/Noconstraints"


#%% save tabseparateddata TR

embedingpathdir = '/home/elamouri/TestsForPaper/TestingRUN_MULTI_FinalModelCorrected/'
type_='alldata'
kind='alpha'
# kind='Noconstraints'
# embedingpathdirs = glob(f'{embedingpathdir}/**/**/**/Em*/*')
# embedingpathdirs = glob(f'{embedingpathdir}/**/{kind}*/**/Em*/')
embedingpathdirs = glob(f'{embedingpathdir}/**/{kind}*/**/Em*/')
out='/home/elamouri/TestsForPaper/dataset_TSCC_MULTI_Final' #dataset_LDPS_Multi' #dataset_TSCC_MULTI'
datasets='ArticularyWordRecognition BasicMotions ERing HandMovementDirection Handwriting FingerMovements Epilepsy'.split(' ')
for dataset in datasets: 
        embedingDatasetpath = []
        for i in embedingpathdirs:
            if f'/{dataset}/' in i:# and dataset in i:ls 
                embedingDatasetpath.append(i)
        for i in embedingDatasetpath:
            if 'alpha' in i:
                fr = i.split('/')[6].split('_')[5]
                nfr = i.split('/')[6].split('_')[7]
                Outpath=f"{out}/{dataset}_fr_{fr}_nfr_{nfr}/{type_}/"
            else:
                Outpath=f"{out}/{dataset}_Noconstraints/{type_}/"
            print(Outpath)
            emb = np.load(f'{i}TransformedData_False_Alldata_True.npy')
            print(f'EMB siz: {dataset} {emb.shape}')
            # break;continue
            if 'alpha' in i:
                tabsaperateddata(f'{dataset}_fr_{fr}_nfr_{nfr}', emb, Outpath=Outpath) # 
            else:
                tabsaperateddata(f'{dataset}_Noconstraints', emb, Outpath=Outpath)
 #%% save tabserpateddata IN setting univariate
# embedingpathdir = "/home/elamouri/TestsForPaper/TestingRunTrainData/MUL"
# embedingpathdir = '/home/elamouri/CDPS/TestingRUN_MULTI_IN/MUL'
embedingpathdir = '/home/elamouri/TestsForPaper/TestingRUN_MULTI_IN_FinalModelCorrected/'

type_='test'
kind='alpha'
# kind='Noconstraints'
embedingpathdirs = glob(f'{embedingpathdir}/**/{kind}*/**/Em*/')
# embedingpathdirs = glob(f'{embedingpathdir}/**/No*/**/Em*/')
# out='/home/elamouri/TestsForPaper/dataset_TSCC_MULTI_Final' #dataset_LDPS_Multi' #dataset_TSCC_MULTI'
# datasets='ArticularyWordRecognition BasicMotions ERing HandMovementDirection Handwriting FingerMovements Epilepsy'.split(' ')
out = f"/home/elamouri/TestsForPaper/dataset_All_Uni/{dataset}/{types}/"
for dataset in datasets: 
        embedingDatasetpath = []
        for i in embedingpathdirs:
            if f'/{dataset}/' in i and kind in i:
                embedingDatasetpath.append(i)
        # print(f'{dataset} : {len(embedingDatasetpath)}')
        for i in embedingDatasetpath:
            if 'alpha' in i:
                # fr = i.split('/')[7].split('_')[5]
                # nfr = i.split('/')[7].split('_')[7]
                fr = i.split('/')[6].split('_')[5]
                nfr = i.split('/')[6].split('_')[7]
# 
                Outpath=f"{out}/{dataset}_fr_{fr}_nfr_{nfr}/{type_}/"
            else:
                Outpath=f"{out}/{dataset}_Noconstraints/{type_}/"
            print(Outpath)
            embTrain = np.load(f'{i}TransformedData_False_Train.npy')
            embTest = np.load(f'{i}TransformedData_False_Test.npy')
            if type_=='test':
                emb = embTest
            else:
                emb = np.concatenate([embTrain,embTest])
            # continue
            if 'alpha' in i:
                tabsaperateddata(f'{dataset}_fr_{fr}_nfr_{nfr}', emb, Outpath=Outpath) # 
            else:
                tabsaperateddata(f'{dataset}_Noconstraints', emb, Outpath=Outpath)
        


def saveconstraints(constraintspath, frs, nfrs, Outpath)
    for fr in frs:
        for nfr in nfrs:
            for type_ in ['Alldata', 'Traindata']:
                for dataset in datasets: 
                    if type_ == 'Alldata':
                        types = 'alldata'
                    elif type_ == 'Traindata':
                        types = 'train'
                    constrainpath = f'{constraintspath}/{dataset}_{type_}'
                    ML = np.load(f'{constrainpath}/{dataset}_{type_}_fr_{fr:.2f}_ML_tn_{nfr}.npy')
                    CL = np.load(f'{constrainpath}/{dataset}_{type_}_fr_{fr:.2f}_CL_tn_{nfr}.npy')
                    tabserpatedconstraitns(dataset, ML, CL, fr, nfr, Outpath)