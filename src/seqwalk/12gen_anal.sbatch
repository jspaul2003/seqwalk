#!/usr/bin/bash
#SBATCH --job-name=12gen_anal
#SBATCH --output=12gen_anal.%j.out
#SBATCH --error=12gen_anal.%j.err
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c 32
#SBATCH --mem=32GB
python -u analysis2.py
