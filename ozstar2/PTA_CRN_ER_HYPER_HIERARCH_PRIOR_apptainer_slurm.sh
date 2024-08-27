#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=5gb

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

touch "job_trackers/ptmcmc_PTA_CRN_ER_HP_${1}"

echo "job_trackers/ptmcmc_PTA_CRN_ER_HP_${1}"

apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_DJRfix.sif python /home/mmiles/soft/GW/ozstar2/CRN_PTMCMC_HYPERMODEL_HierarchPrior.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/pta_objects/MPTA_CRN_ER.pkl -results CRN_ER_HP_run_$1 -alt_dir out_ptmcmc/PTA_RUN_HIERARCH_PRIOR/

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

rm -f "job_trackers/ptmcmc_PTA_CRN_ER_HP_${1}"


echo done
