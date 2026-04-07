#!/bin/bash
#SBATCH --job-name=shortcut-mrsi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=v100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.marshall@yale.edu
#SBATCH --output=logs/shortcut-mrsi-fold-1%j.out
#SBATCH --error=logs/shortcut-mrsi-fold-1%j.err

module load miniconda
eval "$(conda shell.bash hook)"

conda activate mrsi-shortcut

cd /gpfs/gibbs/pi/duncan/am3968/MRSI_Project/shortcut-mrsi-main

nvidia-smi

python main.py \
    --data_path "/gpfs/gibbs/pi/duncan/am3968/MRSI_Project/shortcut-mrsi-main/data_processed" \
    --train_patients '14, 15, 16, 17, 18, 19, 20, 22, 79, 74, 78, 81, 84, 86, 91, 93, 96, 98, 100' \
    --valid_patients '8, 9, 11, 12' \
    --test_patients '1, 2, 4, 6' \
    --save_dir 'checkpoints-bootstrap-2' \
    --batch_size 8 \
    --num_epochs 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --bootstrap_every 2 \
    --denoise_timesteps 128 \
    --scheduler cosine \
    --lr_min 1e-6 \
    --grad_clip 1.0 \
    --ema_decay 0.9999 \
    --device cuda \
    --num_workers 4 \
    --seed 42 \
    --save_every 100 \
    --sample_every 25