#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=8:00:00
#SBATCH --account=p_ml_cv
#SBATCH --output=%j.out
#SBATCH --error=%j.err

echo "++++++++++ resources granted"

cd ..  # go to root folder
pwd

source ./tools/requirements.sh
echo "++++++++++ requirements ok"

source ./tools/run.sh
echo "++++++++++ done!"

exit 0