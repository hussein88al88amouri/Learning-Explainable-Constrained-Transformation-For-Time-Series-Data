#!/bin/bash
. job_scripts/functions.sh
#setting the correct python environment
#source ~/home/anaconda3/bin/activate
#conda activate py38

#To limit the number of scripts runing in parallel
n_proc=1 # Number of processes to execute in parallel, -1 unlimited
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

echo -e "\n Runing with NoConstraints set\n\n"
#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves" #CBF
#datasets="ArticularyWordRecognition BasicMotions ERing HandMovementDirection Handwriting FingerMovements Epilepsy FaceDetection"
datasets="FaceDetection"
nkiter=500
bsize=64
bCsize=16
learning_rate=0.01

min_shapelet_length=0.15
ratio_n_shapelets=10
shapelet_max_scale=3

citer="" 
Savescore="--Savescore" #"" if you don't want to save clustering tests
Alldata="--Alldata" #"" if you want to seprate between them
disable_cuda= #"--disable_cuda " #"" if you don't want cuda
cuda_ID=0

Imp="MUL"
ple=10
checkptsaveEach=40
# CheckpointModel don't give this param only if u give the absolute path for the model, it will automatically search for a checkpoint model to continue from

Outdir=~/CDPS/TestingRUN_MULTI
dataPath=~/Multivariate_ts

logdir=${Outdir}/logsNocon
mkdir -p ${logdir}
for dataset in $datasets; do
                LoadInitShapeletsBeta=$(echo ${Outdir}/${Imp}/${dataset}/InitialzationModel)
                limitjobs
                { 
		   label=Nocon_${dataset}
		   outfile=$(find_next_run ${logdir} ${label})
                   python3 ~/CDPS/bin/Train_CLDPS_pytorch.py ${Outdir} ${dataset} ${dataPath} \
                        --Imp ${Imp} --ple ${ple} --min_shapelet_length ${min_shapelet_length} --ratio_n_shapelets ${ratio_n_shapelets} \
                        --shapelet_max_scale ${shapelet_max_scale} --nkiter ${nkiter} --bsize ${bsize} --learning_rate ${learning_rate} \
                        --LoadInitShapeletsBet ${LoadInitShapeletsBeta} --checkptsaveEach ${checkptsaveEach} --cuda_ID ${cuda_ID}\
                         ${Savescore} ${Alldata} ${disable_cuda} ${citer} \
                       >> ${logdir}/${label}/Nocon_${dataset}.log  2>&1 
                } & 

done

for j in `jobs -rp`; do
    wait $j
done
