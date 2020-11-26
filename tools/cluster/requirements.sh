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
module load h5py/2.10
