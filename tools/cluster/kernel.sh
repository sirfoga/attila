srun --pty -p ml -n 1 -c 2 --gres=gpu:1 --mem-per-cpu 2583 -t 00:30:00 bash -l -A p_ml_cv

KERNEL_NAME="attila"
KERNELS_DIR=/home/${USER}/kernels/

mkdir -p ${KERNELS_DIR}

module load PythonAnaconda/3.7

# optional: load the following to test if packages conflict
# module load Keras
# module load TensorFlow
# module load scikit-learn

conda create --prefix ${KERNELS_DIR}/${KERNEL_NAME} python=3.7.4
source activate ${KERNELS_DIR}/${KERNEL_NAME}
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