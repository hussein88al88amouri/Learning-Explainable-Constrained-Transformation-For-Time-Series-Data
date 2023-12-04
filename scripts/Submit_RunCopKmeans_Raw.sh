#!/bin/bash
. job_scripts/functions.sh

#----------------------------------------------------------------------#

run=RawCopKmeans_trainConstraints
datasetArchive=Multivariate_ts
constraintdirname=NN_constraints_Genci
Alldata="" #"--Alldata"
Outdirname=${run}_results

#----------------------------------------------------------------------#
RootPath=/home2020/home/icube/elamouri/WorkSpace

scriptdir=${RootPath}/Projects/CDPS/scripts
scriptname=Sbatch_RunCopKmeans_Raw.sh

dataBasePath=${RootPath}/Datasets
dataPath=${dataBasePath}/${datasetArchive}

modelBasePath=${RootPath}/Projects/CDPS/output
Outdir=${modelBasePath}/${Outdirname}

constraintdir=${RootPath}/Projects/CDPS/${constraintdirname}

LogsErrorsBasePath=${RootPath}/Projects/CDPS/output/EOLogs
logBasePath=${LogsErrorsBasePath}/logs
errorBasePath=${LogsErrorsBasePath}/errors
logdir=${logBasePath}/${run}
errordir=${errorBasePath}/${run}
mkdir -p logdir
mkdir -p errordir

#typeoftrain=${run}
# Genci
#gpu=v100-32g
#module load tensorflow-gpu/py3/2.5.0

#----------------------------------------------------------------------#

datasets="EthanolConcentration Libras Epilepsy BasicMotions MotorImagery FaceDetection UWaveGestureLibrary Handwriting PhonemeSpectra Heartbeat RacketSports Cricket AtrialFibrillation StandWalkJump FingerMovements NATOPS HandMovementDirection ArticularyWordRecognition EigenWorms PenDigits PEMS-SF SelfRegulationSCP1 SelfRegulationSCP2 DuckDuckGeese LSST ERing"
#datasets="ERing"
datasets="Libras Epilepsy BasicMotions UWaveGestureLibrary Handwriting RacketSports Cricket ArtrialFibrillation StandWalkjump NATOPS ArticularyWordRecognition PenDigits SelfRegulationSCP1 ERing"
fr_="0.05 0.15 0.25"
#fr_="0.05"
nfr="0 1 2 3 4 5 6 7 8 9"
#nfr="0"

DataType=Raw
type=independent
initialization=random
metric=dtw_distance
max_iter=100
trial=1
CDPSEmbPath=-1
rep=10
#----------------------------------------------------------------------#

echo Submiting ${run}

if [ $constraintdirname == None ]; then
  fr_="0"
  nfr="0"
  rep=1
fi
for fr in ${fr_}; do
   for nf in ${nfr}; do   
      for dataset in $datasets; do
        for (( i=1; i<=$rep; i++ )); do
	  trialNumber=${i}
	  
	  label=RCkm_${dataset}_${fr}_${nf}_${i}
	  outfile=$(find_next_run ${logdir} ${label})
	  f="$(basename -- $outfile)"
	  mkdir -p ${errordir}/${label}
	  errorfile=${errordir}/${label}/${f}

	  echo sbatch --job-name=${label} --output=${outfile}.out --error=${errorfile}.out ${scriptdir}/${scriptname} ${label} ${dataset} ${Outdir} ${dataPath} ${logdir} ${fr} ${nf} ${constraintdir} ${type} ${DataType} ${metric} ${trial} ${trialNumber} ${max_iter} ${initialization} ${CDPSEmbPath} ${Alldata}
	  RES=$(sbatch --job-name=${label} --output=${outfile}.out --error=${errorfile}.out ${scriptdir}/${scriptname} ${label} ${dataset} ${Outdir} ${dataPath} ${logdir} ${fr} ${nf} ${constraintdir} ${type} ${DataType} ${metric} ${trial} ${trialNumber} ${max_iter} ${initialization} ${CDPSEmbPath} ${Alldata} )

	  echo ${RES}
	  [ -e ${outfile}.id ] && rm ${outfile}.id
	  echo ${RES##* } > ${outfile}.id
	    
	  done
      done
    done
done
