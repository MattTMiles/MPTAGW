#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH -o %x.out
#SBATCH --mem-per-cpu=3560MB
#SBATCH --tmp=3gb
#SBATCH --propagate=STACK

source ~/.bashrc
#export OMP_NUM_THREADS=1
ml conda

ml gcc/12.2.0
ml openmpi/4.1.4


conda activate mpippcgw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

touch "PTA_CRN_PL_MISSPEC_HYPERMODEL_PTMCMC_${1}"

echo "PTA_CRN_PL_MISSPEC_HYPERMODEL_PTMCMC_${1}"
ulimit -s 16384

i=0
while [ $i -lt 1 ]; do
    #mpirun -np 8 python /home/mmiles/soft/GW/ozstar2/enterprise_CRN.py -partim /fred/oz002/users/mmiles/MPTA_GW/partim/ -results CRN_ER_run_$1 -noise_search pl_nocorr_freegam extra_red -sampler ptmcmc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json -alt_dir out_ptmcmc/PTA_RUN/ -psrlist /fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list
    python /home/mmiles/soft/GW/ozstar2/CRN_PTMCMC_HYPERMODEL.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/pta_objects/PTA_CRN_MISSPEC_RED_DM_FULL.pkl -results CRN_ER_run_$1 -alt_dir out_ptmcmc/PTA_RUN/HYPER_MISSPEC_RED_DM/
    if [[ "$?" -eq 139 ]]; then
        echo "segfault !";
        rm /fred/oz002/users/mmiles/MPTA_GW/partim/core*
    else echo "no segfault !";
        ((i++));
    fi;
done

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

rm -f "PTA_CRN_PL_MISSPEC_HYPERMODEL_PTMCMC_${1}"

echo done
