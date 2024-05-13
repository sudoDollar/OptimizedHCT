#!/bin/bash
#
#SBATCH --job-name=hpml_project_quant_3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=%x.out


module purge
module load anaconda3/2020.07
pip install dill
pip install einops

cd /scratch/gk2657/project_final/OptimizedHCT/quantized_dynamic

python main.py