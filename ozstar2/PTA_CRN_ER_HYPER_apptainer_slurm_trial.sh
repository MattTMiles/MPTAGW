#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=0:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=12gb

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

touch "job_trackers/ptmcmc_PTA_CRN_ER_${1}"

echo "job_trackers/ptmcmc_PTA_CRN_ER_${1}"

apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer.sif python /home/mmiles/soft/GW/ozstar2/CRN_PTMCMC_HYPERMODEL.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/pta_objects/CRN_freegam_ER_DJR.pkl -results CRN_ER_run_test_$1 -alt_dir out_ptmcmc/PTA_RUN/

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

rm -f "job_trackers/ptmcmc_PTA_CRN_ER_${1}"


echo done
