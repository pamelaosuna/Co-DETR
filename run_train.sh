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
# python -u xx.py
# sh tools/slurm_train.sh \
#  train \
#  train_1 \
#  projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py \
#  works_dir/co-defdetr/

set -x

PARTITION=train
JOB_NAME=train_1
CONFIG=projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py
WORK_DIR=works_dir/co-defdetr/
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=1
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
    # ${SRUN_ARGS} \
python -u tools/train.py ${CONFIG} --work-dir ${WORK_DIR} --launcher "slurm" -p ${PARTITION} \
 --gpu-id 0

echo "job finished"
date