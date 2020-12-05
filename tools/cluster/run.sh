conda deactivate
source activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
echo "using $(which python)"  # just to check 

python experiments.py
