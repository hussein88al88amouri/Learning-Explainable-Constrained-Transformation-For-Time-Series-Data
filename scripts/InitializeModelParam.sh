#!/bin/bash
# bash code to run the sensitivity tests
. job_scripts/functions.sh
#source ~/home/anaconda3/bin/activate
#conda activate py38


n_proc=10 # Number of processes to execute in parallel, -1 unlimited
trap "trap - SIGTERM && kill -- -$$ && exit" SIGINT SIGTERM EXIT
function limitjobs {
        if [ "${n_proc}" != "-1" ]
        then
        echo `jobs -rp`         
                while [ `jobs -rp | wc -l` -ge ${n_proc} ]
                do
                        sleep 10
                done
        fi
}


echo -e "\nRuning model Initialization\n\n"
#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves" #CBF
#datasets="Trace"
#datasets="Adiac BME Beef BirdChicken CBF Car Coffee CricketX CricketY CricketZ ECG200 MoteStrain FaceFour FiftyWords Fish Fungi GunPoint GunPointAgeSpan Herring Lightning2 Lightning7 Meat OSULeaf Plane PowerCons Symbols SyntheticControl Trace"
datasets="ArticularyWordRecognition BasicMotions ERing HandMovementDirection Handwriting FingerMovements Epilepsy" # FaceDetection"
#datasets="DuckDuckGeese"

min_shapelet_length=0.15
ratio_n_shapelets=10
shapelet_max_scale=3

Imp="MUL"
Alldata="--Alldata"
disable_cuda= #"--disable_cuda "
cuda_ID=""




Outdir=~/CDPS/TestingRUN_MULTI_NewTest_dtwPython
dataPath=~/Multivariate_ts #~/Univariate_ts

Constraintsdir=~/CDPS/NN_Constraints
logdir=${Outdir}/logsInitModel
mkdir -p ${logdir}
#--cuda_ID ${cuda_ID} \
for dataset in $datasets; do
        echo "Running $dataset"        
        limitjobs
        { 
	   label=MInit_${dataset}
	   outfile=$(find_next_run ${logdir} ${label})
            python3 ~/CDPS/bin/Train_CLDPS_pytorch.py ${Outdir} ${dataset} ${dataPath} --InitShapeletsBeta \
            --Imp ${Imp} \
            --min_shapelet_length ${min_shapelet_length} --ratio_n_shapelets ${ratio_n_shapelets} --shapelet_max_scale ${shapelet_max_scale} \
            ${Alldata} ${disable_cuda} ${Savescore} \
            >> ${logdir}/${label}/MInit_${dataset}.log  2>&1
        } & 
done

for j in `jobs -rp`; do
    wait $j
done
