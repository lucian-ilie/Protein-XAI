#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:t4:1
#SBATCH --output=survey_embeddings.out

module load python/3.10
source ENV/bin/activate

python embeddings.py bert
python embeddings.py t5
python embeddings.py ankh