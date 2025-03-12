#!/bin/bash
#SBATCH --job-name=neural_net
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# #SBATCH --exclusive
#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --output=neural_net_output.txt
#SBATCH --error=neural_net_error.txt
#SBATCH --account=mpcs51087

srun ./neural_net_gpu_cublas
