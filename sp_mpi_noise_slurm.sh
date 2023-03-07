#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH --job-name=%x.out
#SBATCH --mem=2gb
#SBATCH --tmp=2gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}_PM_WN_SW_REMOVE_DM"

echo "${1}_PM_WN_SW_REMOVE_DM"

mpirun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results PM_WN_SW_REMOVE_DM -noise_search efac ecorr sw -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_WN_models.json -nlive 200 -alt_dir out_ppc/PM_WN/$1

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}_PM_WN_SW_REMOVE_DM"


echo done
