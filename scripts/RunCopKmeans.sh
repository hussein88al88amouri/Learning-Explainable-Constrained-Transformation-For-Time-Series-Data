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

echo -e "\nRuning CopKmeans Clusteringn\n"
#datasets="Trace CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves"
#datasets="CBF"
#datasets="ArticularyWordRecognition BasicMotions ERing HandMovementDirection Handwriting FingerMovements Epilepsy"
#datasets="Adiac BME Beef BirdChicken CBF Car Coffee CricketX CricketY CricketZ ECG200 MoteStrain FaceFour FiftyWords Fish Fungi GunPoint GunPointAgeSpan Herring Lightning2 Lightning7 Meat OSULeaf Plane PowerCons Symbols SyntheticControl Trace TwoPatterns ShapesAll SwedishLeaf ECG5000 ElectricDevices FaceAll FordA FordB Crop Mallat FacesUCR Wafer Rock ScreenType UWaveGestureLibraryX Phoneme"
#datasets="BasicMotions ERing HandMovementDirection Handwriting FingerMovements Epilepsy ArticularyWordRecognition"
datasets="BME"
Alldata="--Alldata"
fr_="0.05 0.15 0.25"
nfr_="0 1 2 3 4 5 6 7 8 9"

Outdir=~/CDPS/CopKmeansDBA_independent${1}${2}
constraintspat=~/CDPS/NN_Constraints
dataPath=~/Univariate_ts
logdir=${Outdir}/logsGenCon
type='dependent'
mkdir -p ${logdir}
for dataName in $datasets; do
 for fr in $fr_; do
    for nfr in $nfr_; do
        echo "Running $dataName" 
        limitjobs
        { 
        python  ~/CDPS/bin/runCopKmeans.py ${Outdir} ${dataName} ${dataPath} --fr ${fr} --nfr ${nfr} --Constraints ${constraintspat} --type ${type} ${Alldata} 
        } & # >> ${logdir}/${label}/DiffGenCon_${dataset}.log  2>&1
    done
  done
done


for j in `jobs -rp`; do
    wait $j
done
