#!/bin/bash

# Directory containing the Python script
SCRIPT_NAME="01_train_fff_celeba.py"
SCRIPT_NAME="02_train_fff_ddp_celeba.py"

LOG_DIR="/home/adam2392/projects/logs/"


# Change to the directory containing the script
# cd "$SCRIPT_DIR"

# Define the training seeds to match np.linspace(1, 10000, 11, dtype=int)
# training_seeds=(1 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
# Define the training seeds from 1 to 100
training_seeds=($(seq 4 4))

# TRAINING_SEED=$(expr ${training_seeds[$i]} \* 20)
TRAINING_SEED=0

# Calculate the GPU index to use for this job
# GPU_INDEX=$(((({TRAINING_SEED[$i]}) % $NUM_GPUS) + 1))
GPU_INDEX=1

# Number of GPUs available
NUM_GPUS=3

# Specify the GPUs to use
GPU_INDICES="5,6,7"  # Adjust this as per available GPUs and your requirement

# Set the environment variable for the GPU
# export CUDA_VISIBLE_DEVICES=$GPU_INDEX,$((GPU_INDEX + 1))
# export CUDA_VISIBLE_DEVICES=$GPU_INDEX

# Construct the command to run the Python script with the current training seed
CMD="python3 $SCRIPT_NAME" # --seed $TRAINING_SEED --log_dir $LOG_DIR"
LOG_FILE="celeba_fff_resnet_batch512_gradaccum_latentdim48_beta10_v1_${SCRIPT_NAME}_gpuindex_${GPU_INDEX}.log"

# Optionally, you can use a job scheduler like `nohup` to run the command in the background
# or `&` to run the command in the background
# export CUDA_VISIBLE_DEVICES=$GPU_INDICES
# CMD="torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_NAME" # --seed $TRAINING_SEED --log_dir $LOG_DIR"
# LOG_FILE="celeba_fff_resnet_batch512_gradaccum_latentdim48_beta100_v1_${SCRIPT_NAME}_multigpu.log"
nohup $CMD > $LOG_FILE 2>&1 &

# echo $TRAINING_SEED
echo "CUDA visible devices is $CUDA_VISIBLE_DEVICES"
echo "Submitted job for GPU index: $GPU_INDEX for script: $SCRIPT_NAME"
echo $CMD
