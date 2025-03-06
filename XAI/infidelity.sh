#!/bin/bash
#SBATCH --job-name=infid_bert
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:t4:1
#SBATCH --output=survey_infid_bert_2ccif.out

module load python/3.10
source ENV/bin/activate
python infidelity.py bert Saliency 2cciF HHASPRKQGKKENGPPHSHTLKGRRLVFDN