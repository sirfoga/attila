#!/bin/bash

#SBATCH --partition=ml
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00
#SBATCH --account=p_ml_cv
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source ./requirements.sh
source ./run.sh

exit 0
