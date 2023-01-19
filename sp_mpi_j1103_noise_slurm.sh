#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --job-name=%x.out
#SBATCH --mem=2gb
#SBATCH --tmp=2gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/J1103_kookaburra/gw_limit_comp/GW_search

touch "${1}_full_freq_WN"

echo "${1}_full_freq_WN"

mpirun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results full_freq_WN -noise_search efac equad ecorr -sampler ppc -partim /fred/oz002/users/mmiles/J1103_kookaburra/gw_limit_comp/GW_search/full_freq -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_WN_models.json -nlive 400 -alt_dir /fred/oz002/users/mmiles/J1103_kookaburra/gw_limit_comp/GW_search/gw_comp/j1103

cd /fred/oz002/users/mmiles/J1103_kookaburra/gw_limit_comp/GW_search
rm -f "${1}_full_freq_WN"


echo done
