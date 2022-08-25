#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=MPTA_GW_search_varyGamma
#SBATCH --mem=15gb
#SBATCH --tmp=15gb


source ~/.bashrc


touch "MPTA_GW_search_varyGamma"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

srun python /home/mmiles/soft/GW/enterprise_simple_varyGamma.py

rm -f "MPTA_GW_search_varyGamma"


echo done
