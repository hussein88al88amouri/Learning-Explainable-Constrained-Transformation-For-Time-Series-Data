#! /bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node 8
#SBATCH --mem=65536

#SBATCH -t 12:00:00

#SBATCH -p publicgpu
#SBATCH --gres=gpu:2

#SBATCH --mail-type=END
#SBATCH --mail-user=lampert@unistra.fr


if [ "$#" -ne 3 ]; then
    echo "submit_script must be called with 3 parameters (the model label, configuration filename, and the gpu model in \{1080, k80, k40, k20\})" >&2
    exit 1
fi


label=$1
configfilename=$2
condaenvironment=$3

jobscriptdir=${HOME}/job_scripts

module load compilers/cuda-9.0
module load python/Anaconda3

source activate ${condaenvironment}

cd ${HOME}/sysmifta

KERAS_BACKEND=tensorflow

python3 check_partially_trained.py ${label} -c ${configfilename}
ret=$?

if [ $ret -ne 0 ]; then
    echo ${label}: training
    echo python3 train_unet.py -l ${label} -c ${configfilename}
    python3 train_unet.py -l ${label} -c ${configfilename}
else
    echo ${label}: continuing training
    echo python3 resume_train_unet.py ${label} -c ${configfilename}
    python3 resume_train_unet.py ${label} -c ${configfilename}
fi

#./experiments_apply_model.sh ${label} ${configfilename} 0

cd ..