#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --job-name=chrom_dm_search
#SBATCH --mem=1gb
#SBATCH --tmp=1gb


source ~/.bashrc


touch "${1}_chrom_dm_search"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

srun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results chrom_dm -noise_search efac_c ecorr_c equad_c chrom dm -sampler bilby -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/total_params.json

rm -f "${1}_chrom_dm_search"


echo done
