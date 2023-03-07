#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --job-name=J1327-0755_PM_WN_SW_REMOVE_ECORR
#SBATCH --mem=5gb
#SBATCH --tmp=5gb


source ~/.bashrc


touch "${1}_PM_WN_SW_REMOVE_ECORR"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enteprise

srun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results WN_SW_REMOVE_ECORR -noise_search efac_c equad_c sw -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/total_params.json -alt_dir

rm -f "${1}_PM_WN_SW_REMOVE_ECORR"


echo done
