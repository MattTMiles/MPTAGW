#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --job-name=MPTA_GW_search_constGamma_SP_priors
#SBATCH --mem=30gb
#SBATCH --tmp=30gb


source ~/.bashrc


touch "MPTA_GW_search_constGamma_SP_priors"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

srun python /home/mmiles/soft/GW/enterprise_simple.py

rm -f "MPTA_GW_search_constGamma_SP_priors"


echo done
