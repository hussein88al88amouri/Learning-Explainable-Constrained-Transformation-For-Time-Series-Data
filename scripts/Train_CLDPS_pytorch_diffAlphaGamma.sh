#!/bin/bash
# bash code to run the sensitivity tests
. job_scripts/functions.sh
#source ~/home/anaconda3/bin/activate
#conda activate py38

#To limit the number of running scripts
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


#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves" #CBF
#datasets="Trace"
datasets="Adiac BME Beef BirdChicken CBF Car Coffee CricketX CricketY CricketZ ECG200 MoteStrain FaceFour FiftyWords Fish Fungi GunPoint GunPointAgeSpan Herring Lightning2 Lightning7 Meat OSULeaf Plane PowerCons Symbols SyntheticControl Trace"
nkiter=500
bsize=32
bCsize=8
learning_rate=0.01

min_shapelet_length=0.15
ratio_n_shapelets=10
shapelet_max_scale=3

citer="" 
Savescore="--Savescore" #"" if you don't want to save clustering tests
Alldata="" #"--Alldata" #"" if you want to seprate between them
disable_cuda= #"--disable_cuda " #"" if you don't want cuda
cuda_ID=1

Imp="MUL_NotAlldata"
ple=5
checkptsaveEach=20
# CheckpointModel don't give this param only if u give the absolute path for the model, it will automatically search for a checkpoint model to continue from


gamma="0 0.25 0.5 0.75 1 1.25 1.5 1.75 2" 
alpha="0 0.25 0.5 0.75 1 1.25 1.5 1.75 2"

fr_=0.30
nfr=0

zNormalize="" #"--zNormalize"
# error logs

Outdir=~/CDPS/TestNN_Not_Norm_New
dataPath=~/Univariate_ts
Constraintsdir=~/CDPS/NN_Constraints
logdir=${Outdir}/logsDiffAlphaGamma_NotAlldata
mkdir -p ${logdir}
for dataset in $datasets; do
    LoadInitShapeletsBeta=$(echo ${Outdir}/${Imp}/${dataset}/InitialzationModel)
    for a in $alpha; do
        for g in $gamma; do
                echo "Running $dataset"            
                limitjobs
                { 
    		   label=DShp_${dataset}_${a}_${g}
		   outfile=$(find_next_run ${logdir} ${label})
                    python3 ~/CDPS/bin/Train_CLDPS_pytorch.py ${Outdir} ${dataset} ${dataPath}  \
		                --LoadInitShapeletsBeta ${LoadInitShapeletsBeta} \
                        --Imp ${Imp} --ple ${ple} --checkptsaveEach ${checkptsaveEach} \
                        --fr ${fr_} --nfr ${nfr} --gamma ${g} --alpha ${a}  --Constraints ${Constraintsdir} \
                        --min_shapelet_length ${min_shapelet_length} --ratio_n_shapelets ${ratio_n_shapelets} --shapelet_max_scale ${shapelet_max_scale} \
                        --bsize ${bsize} --bCsize ${bCsize} \
                        --nkiter ${nkiter} --learning_rate ${learning_rate} \
                        --cuda_ID ${cuda_ID} ${InitShapeletsBeta} ${disable_cuda} ${Savescore} ${Alldata} \
                        ${Alldata} ${disable_cuda} ${Savescore} ${citer} \
			>> ${logdir}/${label}/DiffShp_${dataset}.log  2>&1 
                } & 
	 done
    done
done
for j in `jobs -rp`; do
    wait $j
done
