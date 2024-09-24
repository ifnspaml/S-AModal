#!/bin/bash

#SBATCH --time=08-00
#SBATCH --job-name=focal-sam
#SBATCH --gres=gpu:01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000M
#SBATCH --mail-type=end
#SBATCH --partition=gpu


module load cuda/11.3
source activate samodal

srun python sam_finetune.py -bs 1 -lr 0.00001 -optimizer aw -dataset sailvos -mode samadpt -if_warmup False