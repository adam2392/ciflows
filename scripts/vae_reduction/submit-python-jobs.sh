#!/bin/bash

# Directory containing the Python script
# SCRIPT_NAME="01_train_vae_celeba.py"
SCRIPT_NAME="01_train_vae_ddp_celeba.py"
SCRIPT_NAME="01_train_vae_ddp_mnistv2.py"
SCRIPT_NAME="03_train_nf_ddp_mnistv2.py"
# SCRIPT_NAME="02_encode_celeba.py"
# SCRIPT_NAME="03_train_nf_ddp_celeba.py"
# SCRIPT_NAME="03_train_nf_ddp_celeba.py"

LOG_DIR="/home/adam2392/projects/logs/"

# Change to the directory containing the script
# cd "$SCRIPT_DIR"

# Calculate the GPU index to use for this job
GPU_INDEX=6

# Specify the GPUs to use
GPU_INDICES="0,1,2"  # Adjust this as per available GPUs and your requirement
GPU_INDICES="3,4,5"  # Adjust this as per available GPUs and your requirement
# Number of GPUs available
NUM_GPUS=3

# Set the environment variable for the GPU
# export CUDA_VISIBLE_DEVICES=$GPU_INDEX
# LOG_FILE="causalmnist_nf_exp1_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"
# CMD="python3 $SCRIPT_NAME"
#  --config ./experiment.yml" # --seed $TRAINING_SEED --log_dir $LOG_DIR"

# Optionally, you can use a job scheduler like `nohup` to run the command in the background
# or `&` to run the command in the background
# LOG_FILE="celeba_cyclicbetal1loss_haircolorscm_vaeresnetreduction_batch128_gradaccum_latentdim48_img128_v1_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"
# LOG_FILE="encodings_haircolor_celeba_alldata_cyclicbeta_noimageaug_vaeresnetreduction_batch1024_norm01_latentdim48_img128_v1cont_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"

export CUDA_VISIBLE_DEVICES=$GPU_INDICES
LOG_FILE="causalmnist_nf_exp3_${SCRIPT_NAME}_${NUM_GPUS}gpus.log"
CMD="torchrun --master_port=29502 --nproc_per_node=$NUM_GPUS $SCRIPT_NAME" # --seed $TRAINING_SEED --log_dir $LOG_DIR"

# LOG_FILE="celeba_vaeresnetreduction_batch1024_latentdim48_img128_v1_${SCRIPT_NAME}_seed_${GPU_INDEX}.log"
nohup $CMD > $LOG_FILE 2>&1 &

echo "CUDA visible devices is $CUDA_VISIBLE_DEVICES"
echo "Submitted job for GPU index: $GPU_INDEX for script: $SCRIPT_NAME"
echo $CMD