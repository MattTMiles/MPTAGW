#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH -o %x-%j.out
#SBATCH --mem=4gb
#SBATCH --tmp=4gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}_${2}"

echo "${1}_${2}"

mpirun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results $2 -noise_search $3 -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/WN_params_update.json -pool 1

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}_${2}"


echo done
