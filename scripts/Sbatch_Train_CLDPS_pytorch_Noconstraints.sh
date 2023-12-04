#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpua100|gpurtx6000|gpurtx5000|gpuv100|gpu1080|gpup100"
#SBATCH -p publicgpu

module load cuda/cuda-11.2

source ~/home/anaconda3/bin/activate
echo conda activate py38
conda activate py38

label=$1
dataset=$2
Outdir=$3
dataPath=$4
logdir=$5

Imp=$6
Alldata=$7

bsize=64
bCsize=16
learning_rate=0.01

min_shapelet_length=0.15
ratio_n_shapelets=10
shapelet_max_scale=3

citer="" 
Savescore="--Savescore" #"" if you don't want to save clustering tests
disable_cuda= #"--disable_cuda " #"" if you don't want cuda
cuda_ID=0

nkiter=6
ple=2
checkptsaveEach=3

LoadInitShapeletsBeta=$(echo ${Outdir}/${Imp}/${dataset}/InitialzationModel)
python3 ~/CDPS/bin/Train_CLDPS_pytorch.py ${Outdir} ${dataset} ${dataPath} \
--Imp ${Imp} --ple ${ple} --min_shapelet_length ${min_shapelet_length} --ratio_n_shapelets ${ratio_n_shapelets} \
--shapelet_max_scale ${shapelet_max_scale} --nkiter ${nkiter} --bsize ${bsize} --learning_rate ${learning_rate} \
--LoadInitShapeletsBet ${LoadInitShapeletsBeta} --checkptsaveEach ${checkptsaveEach} \
--cuda_ID ${cuda_ID} ${Savescore} ${Alldata} ${disable_cuda} ${citer} \
>> ${logdir}/${label}/Noconstraints_${dataset}.log  2>&1