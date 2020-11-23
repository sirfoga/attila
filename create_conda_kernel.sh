KERNEL_NAME="ml-conda-py3.6"
KERNELS_DIR=/home/${USER}/kernels/

mkdir -p ${KERNELS_DIR}

module load modenv/ml
module load PythonAnaconda/3.6
module load Keras/2.2.4-fosscuda-2018b-Python-3.6.6
module load TensorFlow/1.14.0-PythonAnaconda-3.6
conda create --prefix ${KERNELS_DIR}/${KERNEL_NAME} python=3.6
conda activate ${KERNELS_DIR}/${KERNEL_NAME}
conda install ipykernel
python -m ipykernel install --user --name "${KERNEL_NAME}"
# install other packages
conda deactivate
