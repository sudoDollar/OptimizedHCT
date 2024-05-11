#!/bin/bash
#
#SBATCH --job-name=hpml_project_27
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --output=%x.out

export WANDB_API_KEY=<wandb-key>

# directory="./log"
# mv $directory log_old_9
# if [ ! -d "$directory" ]; then
#     mkdir -p "$directory"
# fi

module purge
module load anaconda3/2020.07
source activate /scratch/ar7996/HPML/penv
cd /scratch/ar7996/HPML/project/

python train.py

unset WANDB_API_KEY