#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --job-name=DM_BL_BH_CHROM
#SBATCH --mem=5gb
#SBATCH --tmp=5gb


source ~/.bashrc


touch "${1}_DM_BL_BH_CHROM"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enteprise

srun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results DM_BL_BH_CHROM -noise_search efac_c equad_c ecorr_C dm band_low band_high chrom -sampler bilby -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/total_params.json -pool 1

rm -f "${1}_DM_BL_BH_CHROM"


echo done
