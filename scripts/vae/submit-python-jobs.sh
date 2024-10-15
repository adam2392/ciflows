#!/bin/bash

# Directory containing the Python script
SCRIPT_NAME="01_train_vae.py"
SCRIPT_NAME="02_train_fff_vae.py"
SCRIPT_NAME="03_train_fif_resnet.py"

LOG_DIR="/home/adam2392/projects/logs/"

# Number of GPUs available
NUM_GPUS=8

# TRAINING_SEED=$(expr ${training_seeds[$i]} \* 20)
TRAINING_SEED=0

# Calculate the GPU index to use for this job
# GPU_INDEX=$(((({TRAINING_SEED[$i]}) % $NUM_GPUS) + 1))
GPU_INDEX=0

# Set the environment variable for the GPU
# export CUDA_VISIBLE_DEVICES=$GPU_INDEX,$((GPU_INDEX + 1))
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

# Construct the command to run the Python script with the current training seed
CMD="python3 $SCRIPT_NAME" # --seed $TRAINING_SEED --log_dir $LOG_DIR"

# Optionally, you can use a job scheduler like `nohup` to run the command in the background
# or `&` to run the command in the background
LOG_FILE="fif_latentdim12_beta50_mnist_${SCRIPT_NAME}_seed_${TRAINING_SEED}.log"
nohup $CMD > $LOG_FILE 2>&1 &

echo $TRAINING_SEED
echo "GPU index is $CUDA_VISIBLE_DEVICES"
echo "Submitted job for training seed: $TRAINING_SEED for script: $SCRIPT_NAME"
