#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -o %x.out
#SBATCH --mem=2gb
#SBATCH --tmp=2gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}_${2}"

echo "${1}_${2}"

mpirun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results $2 -noise_search $3 -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/WN_params_inc_extra_ecorr.json -nlive 200 -alt_dir out_ppc/extra_ecorr

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}_${2}"


echo done
