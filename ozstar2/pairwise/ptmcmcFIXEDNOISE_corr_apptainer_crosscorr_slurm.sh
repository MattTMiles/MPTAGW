#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=10000MB

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata


apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_DJRfix.sif python /home/mmiles/soft/GW/ozstar2/pairwise/enterprise_run_cross_corrs_FIXEDNOISE_corr_pbilby.py -pair $1 $2 -results ${1}_${2} -alt_dir /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/corr_runs/${1}_${2}/ -sampler hyper 


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

echo done