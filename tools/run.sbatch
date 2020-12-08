tools/cluster/run.sbatch
#!/bin/bash

#SBATCH --partition=ml
#SBATCH --partition=ml \
#SBATCH --cpus-per-task=2 \
#SBATCH --gres=gpu:1 \
#SBATCH --mem-per-cpu=2048 \
#SBATCH --time=00:15:00 \
#SBATCH --account=p_ml_cv \
#SBATCH --pty bash
#SBATCH --output=%j.out
#SBATCH --error=%j.err

cd ..  # go to root folder
source .tools/requirements.sh
source .tools/run.sh

exit 0