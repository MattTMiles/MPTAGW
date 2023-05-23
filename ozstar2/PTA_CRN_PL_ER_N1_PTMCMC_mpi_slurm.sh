#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH -o %x.out
#SBATCH --mem=3gb
#SBATCH --tmp=3gb


source ~/.bashrc
#export OMP_NUM_THREADS=1
ml conda

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

touch "PTA_CRN_PL_ER_VARYGAMMA_PTMCMC_${1}"

echo "PTA_CRN_PL_ER_VARYGAMMA_PTMCMC_${1}"

i=0
while [ $i -lt 1 ]; do
    mpirun python /home/mmiles/soft/GW/ozstar2/enterprise_CRN_experiment.py -partim /fred/oz002/users/mmiles/MPTA_GW/partim/ -results CRN_ER_run_$1 -noise_search pl_nocorr_freegam extra_red -sampler ptmcmc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json -alt_dir out_ptmcmc/PTA_RUN/ -psrlist /fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list

    if [[ "$?" -eq 139 ]]; then
        echo "segfault !";
        rm /fred/oz002/users/mmiles/MPTA_GW/partim/core*
    else echo "no segfault !";
        ((i++));
    fi;
done

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

rm -f "PTA_CRN_PL_ER_VARYGAMMA_PTMCMC_${1}"

echo done
