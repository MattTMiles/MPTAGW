#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH -o %x.out
#SBATCH --mem-per-cpu=2gb
#SBATCH --tmp=3gb
#SBATCH --propagate=STACK


source ~/.bashrc
#export OMP_NUM_THREADS=1
ml conda

ml gcc/12.2.0
ml openmpi/4.1.4

conda activate mpippcgw
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2
ulimit -s 16384

touch "PTA_CRN_PL_VARYGAMMA_PTMCMC_${1}"

echo "PTA_CRN_PL_VARYGAMMA_PTMCMC_${1}"

i=0
while [ $i -lt 1 ]; do
    python /home/mmiles/soft/GW/ozstar2/CRN_PTMCMC_MPI.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/pta_objects/PTA_CRN.pkl -results CRN_run_$1 -alt_dir out_ptmcmc/PTA_RUN/

    if [[ "$?" -eq 139 ]]; then
        echo "segfault !";
        rm /fred/oz002/users/mmiles/MPTA_GW/partim/core*
    else echo "no segfault !";
        ((i++));
    fi;
done

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

rm -f "PTA_CRN_PL_VARYGAMMA_PTMCMC_${1}"

echo done
