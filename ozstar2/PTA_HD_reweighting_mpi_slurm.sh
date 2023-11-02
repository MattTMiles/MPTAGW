#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH -o %x.out
#SBATCH --mem-per-cpu=35gb
#SBATCH --propagate=STACK

source ~/.bashrc
#export OMP_NUM_THREADS=1
ml conda

ml gcc/12.2.0
ml openmpi/4.1.4


conda activate mpippcgw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

touch "PTA_HD_REWEIGHT_${1}"

echo "PTA_HD_REWEIGHT_${1}"
ulimit -s 16384

python /home/mmiles/soft/GW/ozstar2/HD_likelihood_reweighting.py $1

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

rm -f "PTA_HD_REWEIGHT_${1}"

echo done
