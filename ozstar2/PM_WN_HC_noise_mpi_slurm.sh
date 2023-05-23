#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/job_outfiles/%x.out
#SBATCH --mem=60gb
#SBATCH --tmp=60gb

source ~/.bashrc
export OMP_NUM_THREADS=1
ml conda

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

touch "${1}_${2}"

echo "${1}_${2}"

mpirun python /home/mmiles/soft/GW/ozstar2/enterprise_run.py -pulsar $1 -results $2 -noise_search $3 -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models_for_PM_WN_HC.json -nlive $4 -alt_dir out_ppc/PM_WN_HC/$1

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

rm -f "${1}_${2}"


echo done
