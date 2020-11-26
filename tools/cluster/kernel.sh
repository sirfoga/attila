srun \
--partition=ml \
--nodes=1 \
--tasks=1 \
--cpus-per-task=2 \
--gres=gpu:1 \
--mem-per-cpu=2583 \
--time=01:00:00 \
--account=p_ml_cv \
--pty zsh

KERNEL_NAME="attila"
KERNELS_DIR=/home/${USER}/kernels/

module purge
module load modenv/ml
module load PythonAnaconda/3.7

# optional: load the following to test if packages conflict
module load Keras
module load TensorFlow
module load scikit-learn
module load matplotlib
module load Pillow

conda create --prefix ${KERNELS_DIR}/${KERNEL_NAME} python=3.7.4
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}
which python  # check 
conda install ipykernel
python -m ipykernel install --user --name "${KERNEL_NAME}"

# optional: install other packages (don't need to be in srun)
# conda install tifffile
# conda install scikit-learn
# conda install matplotlib
# conda install tikzplotlib

# optional: try importing packages
# python

conda deactivate
