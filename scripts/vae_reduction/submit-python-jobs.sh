#!/bin/bash

# Directory containing the Python script
# SCRIPT_NAME="01_train_vae_celeba.py"
SCRIPT_NAME="01_train_vae_ddp_celeba.py"
SCRIPT_NAME="01_train_vae_ddp_mnist.py"
# SCRIPT_NAME="02_encode_celeba.py"
# SCRIPT_NAME="03_train_nf_ddp_celeba.py"
# SCRIPT_NAME="03_train_nf_ddp_celeba.py"

LOG_DIR="/home/adam2392/projects/logs/"

# Change to the directory containing the script
# cd "$SCRIPT_DIR"

# Calculate the GPU index to use for this job
GPU_INDEX=0

# Specify the GPUs to use
GPU_INDICES="0,1"  # Adjust this as per available GPUs and your requirement
# GPU_INDICES="3,4,5"  # Adjust this as per available GPUs and your requirement
# Number of GPUs available
NUM_GPUS=2

# Set the environment variable for the GPU
export CUDA_VISIBLE_DEVICES=$GPU_INDEX
# Construct the command to run the Python script with the current training seed
CMD="python3 $SCRIPT_NAME" # --seed $TRAINING_SEED --log_dir $LOG_DIR"

# Optionally, you can use a job scheduler like `nohup` to run the command in the background
# or `&` to run the command in the background
LOG_FILE="mnist_cyclicbetal1loss_vaeresnetreduction_batch128_gradaccum_latentdim48_img32_v2_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"
# LOG_FILE="celeba_cyclicbetal1loss_haircolorscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v1_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"
# LOG_FILE="encodings_haircolor_celeba_alldata_cyclicbeta_noimageaug_vaeresnetreduction_batch1024_norm01_latentdim48_img128_v1cont_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"

# export CUDA_VISIBLE_DEVICES=$GPU_INDICES
# LOG_FILE="celeba_haircolor_nfon_128flows_alldata_cyclicresnetvaereduction_batch1024_latentdim48_hcdim4_trainableedges_sep4and8_v1_${SCRIPT_NAME}_seed_multigpu.log"
# LOG_FILE="celeba_cyclicbetal1loss_haircolorscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v1_${SCRIPT_NAME}_seed_multigpu.log"
# CMD="torchrun --master_port=29501 --nproc_per_node=$NUM_GPUS $SCRIPT_NAME" # --seed $TRAINING_SEED --log_dir $LOG_DIR"

# LOG_FILE="celeba_vaeresnetreduction_batch1024_latentdim48_img128_v1_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"
nohup $CMD > $LOG_FILE 2>&1 &

echo "CUDA visible devices is $CUDA_VISIBLE_DEVICES"
echo "Submitted job for GPU index: $GPU_INDEX for script: $SCRIPT_NAME"
echo $CMD