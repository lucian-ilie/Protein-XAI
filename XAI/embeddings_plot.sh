#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --ntasks-per-node=1
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=survey_plot.out
module load python/3.10
source ENV/bin/activate
python embeddings_plot.py
