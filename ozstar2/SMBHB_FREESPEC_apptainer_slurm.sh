#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=4gb

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

touch "job_trackers/ptmcmc_${1}_${2}"

echo "job_trackers/ptmcmc_${1}_${2}"

apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer.sif python /home/mmiles/soft/GW/ozstar2/enterprise_run_pbilby.py -pulsar $1 -results $2 -noise_search $3 -sampler hyper -partim /fred/oz002/users/mmiles/MPTA_GW/partim_august23_snr10/partim_updated -noisefile /fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_noise_values_SGWBWN_checked.json -alt_dir out_ptmcmc/FREESPEC/$1

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

rm -f "job_trackers/ptmcmc_${1}_${2}"


echo done
