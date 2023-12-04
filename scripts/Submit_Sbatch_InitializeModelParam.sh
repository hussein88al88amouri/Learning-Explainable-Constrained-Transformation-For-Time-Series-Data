#!/bin/bash
. job_scripts/functions.sh


datasets="Trace CBF Beef Adiac BME Coffee CricketX FaceAll FaceFour Lightning7 OSULeaf SwedishLeaf TwoPatterns SyntheticControl FiftyWords Lightning2 UWaveGestureLibraryX ElectricDevices StarLightCurves Wafer Car CricketY CricketZ Crop ECG200 ECG5000 GunPoint Phoneme Meat Mallat Fish FacesUCR BirdChicken FordA FordB Fungi GunPointAgeSpan Herring MoteStrain Plane PowerCons Rock ScreenType Symbols ShapesAll"
#datasets="Trace CBF"
dataPath=~/CDPS/Univariate_ts
Outdir=~/CDPS/TestingRun
Alldata="--Alldata"
Imp="MUL"
logdir=${Outdir}/logsInitParam
#mkdir -p ${Outdir}
var_=0
for dataset in $datasets; do
        #var_=$((var_+1))
        label=InitD_${dataset}
        #_${var_}
        outfile=$(find_next_run ${logdir} ${label})
        echo sbatch -J ${label} -o ${outfile}.out Sbatch_InitializeModelParam.sh ${label} ${dataset} ${Outdir} ${dataPath} ${logdir} ${Imp} ${Alldata}
        #RES=$(sbatch -J ${label} -o ${outfile}.out Sbatch_InitializeModelParam.sh ${label} ${dataset} ${Outdir} ${dataPath} ${logdir} ${Imp} ${Alldata})
        ./Sbatch_InitializeModelParam.sh ${label} ${dataset} ${Outdir} ${dataPath} ${logdir} ${Imp} ${Alldata}
	echo ${RES}
        [ -e ${outfile}.id ] && rm ${outfile}.id
        echo ${RES##* } > ${outfile}.id
done     


