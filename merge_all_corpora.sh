#!/bin/bash
#SBATCH --job-name=merge_all_genotypes
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=centos7
#SBATCH --error=merge_genotype_corpora12.err
#SBATCH --output=merge_genotype_corpora12.out
#SBATCH --partition=sched_any
module purge
module load anaconda3
source activate torch-env
module add python/3.6.3
python3 merge_genotype_corpora.py --corpus-ids 1 2 --pops ASW ASW --corpus-id 12 --append SNPS

