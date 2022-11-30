#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --job-name=%x.out
#SBATCH --mem=2gb
#SBATCH --tmp=2gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}_J1103_filtered_GW_limit"

echo "${1}_J1103_filtered_GW_limit"

mpirun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results filtered_GW_limit_equad -noise_search efac equad ecorr spgwc -sampler ppc -partim /fred/oz002/users/mmiles/J1103_kookaburra/gw_limit_comp/GW_search/filtered -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_WN_models.json -nlive 400 -alt_dir gw_comp/j1103

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}_J1103_filtered_GW_limit"


echo done
