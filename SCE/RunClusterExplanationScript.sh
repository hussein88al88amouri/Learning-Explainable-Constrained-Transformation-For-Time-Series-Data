#!/bin/bash
n_proc=4 # Number of processes to execute in parallel, -1 unlimited
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
source /home/elamouri/.conda/envs/pytorch_cdps/bin/activate pytorch_cdps

echo -e Running Cluster Explanation on:
datasets="ShapesAll OSULeaf Coffee ScreenType Phoneme CBF Rock CricketZ Lightning7 Mallat Car CricketX MoteStrain Fungi Symbols GunPointAgeSpan Lightning2 SwedishLeaf ECG200 Adiac Fish BirdChicken PowerCons BME Plane Meat SyntheticControl Beef FacesUCR GunPoint FaceFour Herring FiftyWords FaceAll CricketY"
datasets="BME CBF ECG200 GunPointAgeSpan GunPoint Herring MoteStrain OSULeaf Plane Symbols"
datapath=/home/elamouri/Univariate_ts
outdir=/home/elamouri/Thesis/Chapter4/ComparisionToPCAReduction
Alldata='--Alldata'
for dataset in ${datasets}; do
    echo ${dataset}
    logdir=${outdir}/${dataset}/logdirv1
    echo logdir ${logdir}
    mkdir -p ${logdir}
#    limitjobs
#    { 
        python ClusterExplanationScript.py ${dataset} ${datapath} ${outdir} ${Alldata} #>> ${logdir}/${dataset}_file.log 2>&1 
#    }&
#    echo python ClusterExplanationScript.py ${dataset} ${datapath} ${outdir} ${Alldata}
done
for j in `jobs -rp`; do
    wait $j
done
