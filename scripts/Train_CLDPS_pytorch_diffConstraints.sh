#!/bin/bash
# bash code to run the sensitivity tests
. job_scripts/functions.sh

#source ~/home/anaconda3/bin/activate
#conda activate py38

n_proc=12 # Number of processes to execute in parallel, -1 unlimited
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


echo -e "\nRuning with differnt constraints set\n\n"
#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves" #CBF
#datasets="Trace"
datasets="ArticularyWordRecognition HandMovementDirection Handwriting FingerMovements ERing Epilepsy"
#datasets="Adiac BME Beef BirdChicken CBF Car Coffee Fungi GunPoint Symbols SyntheticControl Trace"

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
cuda_ID=""


Imp="MUL"
ple=10
checkptsaveEach=100
# CheckpointModel don't give this param only if u give the absolute path for the model, it will automatically search for a checkpoint model to continue from

zNormalize="" #"--zNormalize"
gamma=2.5
alpha=2.5

fr_='0.05 0.15 0.25'
nfr="0 1 2 3 4 5 6 7 8 9"

# error logs

Outdir=~/CDPS/TestingRUN_MULTI_NewTest_dtwPython
dataPath=~/Multivariate_ts
Constraintsdir=~/CDPS/NN_Constraints
logdir=${Outdir}/logsDiffConstarints
mkdir -p ${logdir}
for dataset in $datasets; do
    LoadInitShapeletsBeta=$(echo ${Outdir}/${Imp}/${dataset}/InitialzationModel)
    for fr in $fr_; do
        for nf in $nfr; do
                echo "Running $dataset fr ${fr} nfr  ${nf}" 
                limitjobs
                { 
		   label=DCon_${dataset}_${fr}_${nf}
		   outfile=$(find_next_run ${logdir} ${label})
                    python3 ~/CDPS/bin/Train_CLDPS_pytorch.py ${Outdir} ${dataset} ${dataPath} \
                    --LoadInitShapeletsBeta ${LoadInitShapeletsBeta} \
                    --Imp ${Imp} --ple ${ple} --checkptsaveEach ${checkptsaveEach} \
                    --fr ${fr} --nfr ${nf} --gamma ${gamma} --alpha ${alpha}  --Constraints ${Constraintsdir} \
                    --min_shapelet_length ${min_shapelet_length} --ratio_n_shapelets ${ratio_n_shapelets} --shapelet_max_scale ${shapelet_max_scale} \
                    --bsize ${bsize} --bCsize ${bCsize} \
                    --nkiter ${nkiter} --learning_rate ${learning_rate} \
                    ${Alldata} ${disable_cuda} ${Savescore} ${citer} ${zNormalize} \
			>> ${logdir}/${label}/DiffCon_${dataset}.log  2>&1
               } & 

#                    --cuda_ID ${cuda_ID} \
        done
    done
done

for j in `jobs -rp`; do
    wait $j
done

