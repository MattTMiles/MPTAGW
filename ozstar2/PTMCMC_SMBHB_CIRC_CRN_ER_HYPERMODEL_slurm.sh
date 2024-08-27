#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=6gb

source ~/.bashrc
#export OMP_NUM_THREADS=1

ml gcc/12.2.0
ml openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

touch "job_trackers/MPTA_SMBHB_CIRC_CRN_ER_HYPERMODEL_PTMCMC_${1}"

echo "job_trackers/MPTA_SMBHB_CIRC_CRN_ER_HYPERMODEL_PTMCMC_${1}"


apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_upgraded.sif python /home/mmiles/soft/GW/ozstar2/CRN_PTMCMC_HYPERMODEL.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/pta_objects/MPTA_CW_CRN_ER.pkl -results MPTA_CW_CRN_ER_run_$1 -alt_dir out_ptmcmc/SMBHB_CIRC/


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

rm -f "job_trackers/MPTA_SMBHB_CIRC_CRN_ER_HYPERMODEL_PTMCMC_${1}"

echo done
