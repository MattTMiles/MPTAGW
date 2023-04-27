#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/job_outfiles/%x.out
#SBATCH --mem=15gb
#SBATCH --tmp=15gb


source ~/.bashrc
export OMP_NUM_THREADS=1

ml conda
conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

touch "${1}_${2}_${3}"

echo "${1}_${2}_${3}"

mpirun python /home/mmiles/soft/GW/ozstar2/enterprise_cross_corrs.py -pulsars $1 $2 -results $1_$2 -noise_search single_bin_cross_corr_fixedamp -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json -nlive $3 -alt_dir cross_corrs/fixed_amp_$3

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

rm -f "${1}_${2}_${3}"


echo done
