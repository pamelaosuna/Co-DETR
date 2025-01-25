#!/bin/bash

#SBATCH --job-name=codetr
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00

# export MODULEPATH=$MODULEPATH:/scratch/dldevel/osuna/spack/share/spack/modules/linux-almalinux9-zen2/
# # module avail
# module load cuda/11.8.0-gcc-11.3.1-ru677ys

echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate codetr
which python3
echo "loaded conda environment"
echo "start training..."
date

module load nvidia/cuda/12.3.0
# python -u -m pip install --upgrade pip
# pip install --no-cache-dir --upgrade wheel setuptools
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# requirements.txt

# pip install openmim
mim install mmcv-full==1.6
# pip install .

# pip install yacs black==22.1.0 scipy
# # mmdet==2.25
# pandas
# tqdm
# gdown
# zipfile36
# opencv-python

echo job finished
date