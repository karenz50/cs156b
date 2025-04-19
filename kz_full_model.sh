#!/bin/bash

#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --output=kz_models/full.txt

#SBATCH --time=18:00:00

#SBATCH -J "full lung opacity model"
#SBATCH --mail-user=kzhou@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source ~/.bashrc
conda activate cs156b

python kz_models/lung_opacity/full_lung_opacity.py