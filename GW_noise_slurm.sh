#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=MPTA_GW_search_constGamma
#SBATCH --mem=15gb
#SBATCH --tmp=15gb


source ~/.bashrc


touch "MPTA_GW_search_constGamma"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

srun python /home/mmiles/soft/GW/enterprise_simple.py

rm -f "MPTA_GW_search_constGamma"


echo done
