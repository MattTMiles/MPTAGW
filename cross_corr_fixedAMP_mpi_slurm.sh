#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o %x.out
#SBATCH --mem=2gb
#SBATCH --tmp=2gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}_${2}_${3}"

echo "${1}_${2}_${3}"

mpirun python /home/mmiles/soft/GW/enterprise_cross_corrs.py -pulsars $1 $2 -results $1_$2 -noise_search single_bin_cross_corr_fixedamp -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_WN_models.json -nlive $3 -alt_dir cross_corrs/fixed_amp_$3

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}_${2}_${3}"


echo done
