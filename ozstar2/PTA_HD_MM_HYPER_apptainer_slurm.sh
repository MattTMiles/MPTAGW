#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=15gb

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/

touch "job_trackers/ptmcmc_PTA_HD_DJR_${1}"

echo "job_trackers/ptmcmc_PTA_HD_DJR_${1}"

apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_DJRfix.sif python /home/mmiles/soft/GW/ozstar2/HD_PTMCMC_HYPERMODEL.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/pta_objects/MPTA_HD.pkl -results HD_MM_run_$1 -alt_dir /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs_MM/HD/

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/

rm -f "job_trackers/ptmcmc_PTA_HD_DJR_${1}"


echo done
