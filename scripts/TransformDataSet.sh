#!/bin/bash
# bash code to run the sensitivity tests

source ~/home/anaconda3/bin/activate
conda activate py38

#To limit the number of scripts running in parallel
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

#Need some fixing; actually need some planing

echo -e "\n\n Generating Embedings\n\n"
#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves" #CBF
datasets=$1 #"Trace" 
dataPath=$2 #~/CDPS/Univariate_ts/
Type=$3 #Alldata
modelpath=$4 #
Outdir=$5 #~/CDPS/TestNN/
python3 ~/CDPS/bin/TransformDataSet.py ${datasets} ${dataPath} ${Type} ${modelpath} ${Outdir} 
exit 0
for dataset in $datasets; do
                limitjobs
                { 
                    python3 ~/CDPS/bin/Train_CLDPS_pytorch.py ${Outdir} ${dataset} ${dataPath} ${Resultdir} 
                } & 
done

for j in `jobs -rp`; do
    wait $j
done
