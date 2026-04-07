#!/bin/bash
#SBATCH --job-name=eval-shortcut-mrsi
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.marshall@yale.edu
#SBATCH --output=logs/eval-shortcut-mrsi%j.out
#SBATCH --error=logs/eval-shortcut-mrsi%j.err

module load miniconda
eval "$(conda shell.bash hook)"

conda activate mrsi-shortcut

cd /gpfs/gibbs/pi/duncan/am3968/MRSI_Project/shortcut-mrsi-final

nvidia-smi

python eval-save-results.py