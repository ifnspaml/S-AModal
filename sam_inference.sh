#!/bin/bash

#SBATCH --time=8-00
#SBATCH --job-name=eval_asd_sam
#SBATCH --gres=gpu:01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000M
#SBATCH --mail-type=end
#SBATCH --partition=gpu,gpub

module load cuda/11.3
source activate sam_env1

srun python sam_inference.py --save_path predictions_sailvos.json
