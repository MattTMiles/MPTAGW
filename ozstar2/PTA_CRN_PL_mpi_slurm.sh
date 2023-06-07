#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=128
#SBATCH --time=06:00:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/job_outfiles/%x.out
#SBATCH --mem-per-cpu=2560MB

source ~/.bashrc
export OMP_NUM_THREADS=1
ml conda

conda activate mpippcgw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2


i=0
while [ $i -lt 1 ]; do
    mpirun python /home/mmiles/soft/GW/ozstar2/CRN_PPC_MPI.py -pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/pta_objects/PTA_CRN_ER.pkl -nlive 400 -results CRN_ER_run_N128run -alt_dir out_ppc/PTA_RUN/
#    echo $?
#    if [[ $? -eq 139 ]]; then
#        echo "segfault !";
#    else echo "no segfault !";
#        ((i++));
#    fi;
done


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2



echo done
