#!/usr/bin/bash
#SBATCH --job-name=12gen
#SBATCH --output=12gen.%j.out
#SBATCH --error=12gen.%j.err
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c 32
#SBATCH --mem=32GB
python -u 12gen.py
