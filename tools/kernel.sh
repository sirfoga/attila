srun \
--partition=ml \
--nodes=1 \
--tasks=1 \
--cpus-per-task=2 \
--gres=gpu:1 \
--mem-per-cpu=4096 \
--time=00:30:00 \
--account=p_ml_cv \
--pty bash

source ./requirements.sh

conda create --prefix ${KERNELS_DIR}/${KERNEL_NAME} python=3.7.4
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check 
conda install ipykernel
python -m ipykernel install --user --name "${KERNEL_NAME}"

# optional: install other packages (don't need to be in srun)
# source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
# conda activate ${KERNELS_DIR}/${KERNEL_NAME}
# conda install tifffile
# conda install scikit-learn
# conda install matplotlib
# conda install tikzplotlib

# optional: try importing packages
# python

conda deactivate
