#!/bin/bash
##SBATCH --cpus-per-task=40
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
##SBATCH --gres=gpu:2
##SBATCH --constraint="gpua100|gpurtx6000|gpurtx5000|gpuv100|gpu1080|gpup100"
##SBATCH -p grantgpu
##SBATCH -A g2021a270g
##SBATCH  -p publicgpu

. job_scripts/functions.sh

module purge
module load cuda/cuda-11.2
source /home2020/home/icube/elamouri/home/anaconda3/bin/activate py38
#conda activate 

#----------------------------------------------------------------------#
label=${1}
logdir=${5}
Alldata=${17}

DataName=${2}
DataPath=${4}


Outdir=${3}
Constraintsdir=${8}
 
fr=${6}
nf=${7}

DataType=${10}
type=${9}
initialization=${15}
metric=${11}
max_iter=${14}
trial=${12}
trialNumber=${13}


CDPSEmbPath=${16}

#----------------------------------------------------------------------#

CodeRootPath=/home2020/home/icube/elamouri/WorkSpace/Projects/CDPS/bin

echo Runing COP-Kmeans ${label}:
echo logdir at ${logdir}

echo python ${CodeRootPath}/runCopKmeans.py ${Outdir} ${DataName} ${DataPath} ${DataType} --CDPSEmbPath ${CDPSEmbPath} --max_iter ${max_iter} --fr ${fr} --nfr ${nf} --type ${type} --initialization ${initialization} --metric ${metric} --trial ${trial} --Constraints ${Constraintsdir} --trialNumber ${trialNumber} ${Alldata}

python ${CodeRootPath}/runCopKmeans.py ${Outdir} ${DataName} ${DataPath} ${DataType} --CDPSEmbPath ${CDPSEmbPath} --max_iter ${max_iter} --fr ${fr} --nfr ${nf} --type ${type} --initialization ${initialization} --metric ${metric} --trial ${trial} --Constraints ${Constraintsdir} --trialNumber ${trialNumber} ${Alldata}
