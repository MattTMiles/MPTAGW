#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/job_outfiles/%x.out
#SBATCH --mem-per-cpu=10gb

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs

touch "job_trackers/ptmcmc_PTA_HD_DJR_${1}"

echo "job_trackers/ptmcmc_PTA_HD_DJR_${1}"

apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_upgraded.sif python /home/mmiles/soft/GW/ozstar2/CRN_PTMCMC_HYPERMODEL.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/pta_object/MPTA_HD_ER.pkl -results HD_DJR_run_$1 -alt_dir /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs

rm -f "job_trackers/ptmcmc_PTA_HD_DJR_${1}"


echo done
