#!/bin/bash
#SBATCH --job-name=pisp
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:t4:1
#SBATCH --output=survey_pisp.out

module load python/3.10
source ENV/bin/activate

python pisp_interpretation.py bert
python pisp_interpretation.py t5
python pisp_interpretation.py ankh