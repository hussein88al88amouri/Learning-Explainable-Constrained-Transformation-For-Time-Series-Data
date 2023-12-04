#!/bin/bash
# bash code to run the sensitivity tests

#source ~/home/anaconda3/bin/activate
#conda activate py38

#To limit the number of scripts running in parallel
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


echo -e "\nRuning with differnt constraints set\n\n"
#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves"
#datasets="CBF"
#datasets="ArticularyWordRecognition BasicMotions ERing HandMovementDirection Handwriting FingerMovements Epilepsy FaceDetection"
datasets="AtrialFibrillation EthanolConcentration Cricket EigenWorms"
Alldata="--Alldata"
fr_='--fr 0.05 --fr 0.1 --fr 0.15 --fr 0.2 --fr 0.25 --fr 0.3'
nfr=10

Outdir=~/CDPS/NN_Constraints
dataPath=~/Multivariate_ts
logdir=${Outdir}/logsGenCon
mkdir -p ${logdir}
for dataName in $datasets; do
        echo "Running $dataset" 
        limitjobs
        { 
        python  ~/CDPS/bin/GenerateConstraints.py ${Outdir} ${dataName} ${dataPath} ${fr_} ${Alldata} --nfr ${nfr} 
        } & >> ${logdir}/${label}/DiffGenCon_${dataset}.log  2>&1
done


for j in `jobs -rp`; do
    wait $j
done
