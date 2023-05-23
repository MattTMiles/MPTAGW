#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/job_outfiles/%x.out
#SBATCH --mem=40gb
#SBATCH --tmp=40gb

source ~/.bashrc
export OMP_NUM_THREADS=1
ml conda

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2


i=0
while [ $i -lt 1 ]; do
    mpirun python /home/mmiles/soft/GW/ozstar2/enterprise_CRN.py -partim /fred/oz002/users/mmiles/MPTA_GW/partim/ -results CRN_run -noise_search pl_nocorr_freegam -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json -nlive 200 -alt_dir out_ppc/PTA_RUN/ -psrlist /fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list
#    echo $?
#    if [[ $? -eq 139 ]]; then
#        echo "segfault !";
#    else echo "no segfault !";
#        ((i++));
#    fi;
done


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2



echo done
