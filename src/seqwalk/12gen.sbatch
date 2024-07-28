#!/usr/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=analysis.%j.out
#SBATCH --error=analysis.%j.err
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH -c 32
#SBATCH --mem=32GB
python -u analysis.py 14 8 p library14_8 t14_8
python -u analysis.py 12 7 p library12_7 t12_7
python -u analysis.py 16 8 p library16_8 t16_8 
python -u analysis.py 14 7 p library14_7 t14_7
python -u analysis.py 12 6 p library12_6 t12_6
python -u analysis.py 16 7 p library16_7 t16_7 

