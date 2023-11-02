#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=3000MB

source ~/.bashrc
export OMP_NUM_THREADS=1
ml conda

ml gcc/12.2.0
ml openmpi/4.1.4


conda activate mpippcgw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/best_bets

touch "job_trackers/pbilby_${1}_${2}"

echo "job_trackers/pbilby_${1}_${2}"


mpirun -np 8 python /home/mmiles/soft/GW/ozstar2/enterprise_run_pbilby.py -pulsar $1 -results $2 -noise_search $3 -sampler pbilby -partim /fred/oz002/users/mmiles/MPTA_GW/partim_august23_snr10/partim_updated -noisefile /fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_WN_models.json -nlive 400 -alt_dir out_pbilby/$1


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/best_bets

rm -f "job_trackers/pbilby_${1}_${2}"

echo done
