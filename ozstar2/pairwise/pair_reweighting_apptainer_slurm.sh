#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_NO_ER/job_outfiles/%x.out
#SBATCH --mem-per-cpu=550MB

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

i=0
while [ $i -lt 1 ]; do
    apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_DJRfix.sif python /home/mmiles/soft/GW/ozstar2/pairwise/PAIRWISE_likelihood_reweighting_master_script.py -post_file /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/${1}/master_chain.txt -new_pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_NO_ER/pair_objects/${1}.pkl -old_pta /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/${1}.pkl -outdir /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_NO_ER/${1}/

    if [[ "$?" -eq 139 ]]; then
        echo "segfault !";
        rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/core*
        rm /fred/oz002/users/mmiles/MPTA_GW/partim_frank/pp_8/core*
    else echo "no segfault !";
        ((i++));
    fi;
done


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata


echo done