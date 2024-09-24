#!/bin/bash

#SBATCH --time=08-00
#SBATCH --job-name=asdpred_inference
#SBATCH --gres=gpu:01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000M
#SBATCH --mail-type=end
#SBATCH --partition=gpu,gpub


module load cuda/11.3
source activate sam_env1

srun python sam_inference_asdpreds.py --save_path predictions_asd_predbased_wgt_run3.json