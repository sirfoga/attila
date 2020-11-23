#!/bin/bash

#SBATCH --partition=ml
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:10:00
#SBATCH --account=p_ml_cv
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# interactive version:
srun \
--partition=ml \
--nodes=1 \
--tasks=1 \
--cpus-per-task=1 \
--gres=gpu:1 \
--mem-per-cpu=2048 \
--time=00:30:00 \
--account=p_ml_cv \
--pty zsh

module load modenv/ml
module load modenv/scs5
module load scikit-learn
module load Keras
module load TensorFlow

python main.py

exit 0
