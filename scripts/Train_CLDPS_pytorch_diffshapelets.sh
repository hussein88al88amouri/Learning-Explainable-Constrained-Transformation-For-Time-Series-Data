#!/bin/bash
# bash code to run the sensitivity tests
. job_scripts/functions.sh
#source ~/home/anaconda3/bin/activate
#"conda activate py38

n_proc=5 # Number of processes to execute in parallel, -1 unlimited
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


echo -e "\nRuning with Differnt Shapelets set\n\n"

#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves" #CBF
#datasets="Asiac"
#datasets=" BME Beef BirdChicken CBF Car Coffee CricketX CricketY CricketZ ECG200 MoteStrain FaceFour FiftyWords Fish Fungi GunPoint GunPointAgeSpan Herring Lightning2 Lightning7 Meat OSULeaf Plane PowerCons Symbols SyntheticControl Trace"
datasets="TwoPatterns ShapesAll SwedishLeaf ECG5000 ElectricDevices FaceAll FordA FordB Crop Mallat FacesUCR Wafer Rock ScreenType UWaveGestureLibraryX Phoneme"

nkiter=100
bsize=64
bCsize=16
learning_rate=0.01

citer="" 
Savescore="--Savescore" #"" if you don't want to save clustering tests
Alldata=""  #"--Alldata" #"" if you want to seprate between them
disable_cuda= #"--disable_cuda " #"" if you don't want cuda
cuda_ID=

Imp="MUL_NotAllData"
ple=2
checkptsaveEach=5
# CheckpointModel don't give this param only if u give the absolute path for the model, it will automatically search for a checkpoint model to continue from

gamma=2.5
alpha=2.5

min_shapelet_length=0.15
ratio_n_shapelets="1 2 4 6 8 10"
shapelet_max_scale="2 3 4"


fr_=0.3
nfr=0

# error logs

Outdir=~/CDPS/TestingRun1
dataPath=~/Univariate_ts
Constraintsdir=~/CDPS/NN_Constraints

logdir=${Outdir}/logsDiffShapelets
mkdir -p ${logdir}
for dataset in $datasets; do
    for rns in $ratio_n_shapelets; do
        for sms in $shapelet_max_scale; do
                echo "Running $dataset $bname"            
                limitjobs
                { 
		    label=DShp_${dataset}_${rns}_${sms}
   		    outfile=$(find_next_run ${logdir} ${label})
                    python3 ~/CDPS/bin/Train_CLDPS_pytorch.py ${Outdir} ${dataset} ${dataPath} \
                    --Imp ${Imp} --ple ${ple} --checkptsaveEach ${checkptsaveEach} \
                    --fr ${fr_} --nfr ${nfr} --gamma ${gamma} --alpha ${alpha}  --Constraints ${Constraintsdir} \
                    --min_shapelet_length ${min_shapelet_length} --ratio_n_shapelets ${rns} --shapelet_max_scale ${sms} \
                    --bsize ${bsize} --bCsize ${bCsize} \
                    --nkiter ${nkiter} --learning_rate ${learning_rate} \
                    --cuda_ID ${cuda_ID} \
                    ${Alldata} ${disable_cuda} ${Savescore} ${citer} \
	            >> ${logdir}/${label}/DiffShp_${dataset}.log  2>&1 
                } & 
		echo $! > ${logdir}/${label}/run.id
        done
    done
done

for j in `jobs -rp`; do
    wait $j
done
