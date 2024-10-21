#!/bin/bash

# Directory containing the Python script
SCRIPT_NAME="01_train_basic_models.py"
# SCRIPT_NAME="02_train_approx_models.py"

LOG_DIR="/home/adam2392/projects/logs/"

# Number of GPUs available
NUM_GPUS=8

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

# Set the environment variable for the GPU
# export CUDA_VISIBLE_DEVICES=$GPU_INDEX,$((GPU_INDEX + 1))
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

# Construct the command to run the Python script with the current training seed
CMD="python3 $SCRIPT_NAME" # --seed $TRAINING_SEED --log_dir $LOG_DIR"

# Optionally, you can use a job scheduler like `nohup` to run the command in the background
# or `&` to run the command in the background
LOG_FILE="injflow_twostage_batch1024_gradclip1_mnist_trainableq0_nstepsmse20_v4_${SCRIPT_NAME}_seed_${TRAINING_SEED}.log"
nohup $CMD > $LOG_FILE 2>&1 &

echo $TRAINING_SEED
echo "GPU index is $CUDA_VISIBLE_DEVICES"
echo "Submitted job for training seed: $TRAINING_SEED for script: $SCRIPT_NAME"
