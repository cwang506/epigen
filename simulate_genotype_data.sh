#!/bin/bash
#SBATCH --job-name=simulate_data
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 40000
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=centos7
#SBATCH --error=simulate_data_09_1_big.err
#SBATCH --output=simulate_data_09_1_big.out
#SBATCH --partition=sched_any
module purge
module load anaconda3
source activate torch-env
module add python/3.6.3
python3 simulate_data.py --corpus-id 122 --pop ASW --sim-ids 0 --inds 3000 --snps 1387466 --disease-snps 1162907 808021 59983 89931 22601 377757 305546 524297 1253274 177 --model models/param_model_train.xml
