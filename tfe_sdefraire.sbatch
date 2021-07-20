#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=retrieval          # Name of the job
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --output=sdefraire-output.log # Log-file (important!)
#SBATCH --cpus-per-task=2             # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G             # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                  # Number of GPU's
#SBATCH --time=24:00:00               # Max execution time
#

# Activate your Anaconda environment
conda activate tfe         # CHANGEME

# Run your Python script
cd /home/sdefraire/src # CHANGEME
python densenet.py --training_data /scratch/users/sdefraire/datasets/
