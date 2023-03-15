#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --time=00:55:00
#SBATCH -o %x.out
#SBATCH --mem=40gb
#SBATCH --tmp=40gb


source ~/.bashrc
export OMP_NUM_THREADS=1

ml conda

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}_${2}_${3}"

echo "${1}_${2}_${3}"

mpirun python /home/mmiles/soft/GW/enterprise_cross_corrs.py -pulsars $1 $2 -results ${1}_${2}_ozstar2_4cores -noise_search single_bin_cross_corr -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_WN_models.json -nlive $3 -alt_dir cross_corrs/live_${3}_ozstar2

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}_${2}_${3}"


echo done
