#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --time=06:00:00
#SBATCH -o %x.out
#SBATCH --mem-per-cpu=2gb
#SBATCH --tmp=15gb
#SBATCH --propagate=STACK


source ~/.bashrc
#export OMP_NUM_THREADS=1
ml conda

conda activate gw
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2
ulimit -s 16384

touch "PTA_CRN_PL_VARYGAMMA_PTMCMC_${1}"

echo "PTA_CRN_PL_VARYGAMMA_PTMCMC_${1}"

i=0
while [ $i -lt 1 ]; do
    mpirun -np 8 python /home/mmiles/soft/GW/ozstar2/enterprise_CRN.py -partim /fred/oz002/users/mmiles/MPTA_GW/partim/ -results CRN_run_$1 -noise_search pl_nocorr_freegam -sampler ptmcmc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json -alt_dir out_ptmcmc/PTA_RUN/ -psrlist /fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list

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
