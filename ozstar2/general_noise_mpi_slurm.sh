#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/job_outfiles/%x.out
#SBATCH --mem=20gb
#SBATCH --tmp=20gb

source ~/.bashrc
export OMP_NUM_THREADS=1
ml conda

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

touch "${1}_${2}"

echo "${1}_${2}"

mpirun python /home/mmiles/soft/GW/ozstar2/enterprise_run.py -pulsar $1 -results $2 -noise_search $3 -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json -nlive $4 -alt_dir out_ppc/$5/$1

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

rm -f "${1}_${2}"


echo done
