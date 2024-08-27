#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=64gb

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

touch "job_trackers/ptmcmc_PTA_HD_ER_${1}"

echo "job_trackers/ptmcmc_PTA_HD_ER_${1}"

apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_upgraded.sif python /home/mmiles/soft/GW/ozstar2/CRN_PTMCMC_HYPERMODEL.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/pta_objects/HD_freegam_ER_DJR.pkl -results HD_ER_run_$1 -alt_dir out_ptmcmc/PTA_RUN/

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

rm -f "job_trackers/ptmcmc_PTA_HD_ER_${1}"


echo done
