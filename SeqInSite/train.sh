#!/bin/bash
#SBATCH --job-name=SeqInSite
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:t4:1
#SBATCH --output=survey.out

module load python/3.10
source ENV/bin/activate
python SeqInSite.py bert
python SeqInSite.py t5
python SeqInSite.py ankh