conda deactivate
source activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
echo "using $(which python)"  # just to check 

# rm -rf out/  # clean to be sure

# python train_and_eval.py
# python to_augment_or_not.py
python lose_yourself.py