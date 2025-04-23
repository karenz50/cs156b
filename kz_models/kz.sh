#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --partition=expansion
#SBATCH --output=kz_output.txt

#SBATCH --time=01:00:00

#SBATCH -J "test"
#SBATCH --mail-user=kzhou@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=expansion

source ~/.bashrc
conda activate cs156b

python kz_models/lung_opacity.py