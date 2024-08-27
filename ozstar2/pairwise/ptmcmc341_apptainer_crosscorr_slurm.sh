#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=10000MB

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata
touch "job_trackers/ptmcmc341_${1}_${2}_${3}"
echo "job_trackers/ptmcmc341_${1}_${2}_${3}"



i=0
while [ $i -lt 1 ]; do
    apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_DJRfix.sif python /home/mmiles/soft/GW/ozstar2/pairwise/enterprise_run_cross_corrs_341_pbilby.py -pair $1 $2 -results ${1}_${2}_${3} -alt_dir /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/${1}_${2}/ -sampler hyper 

    if [[ "$?" -eq 139 ]]; then
        echo "segfault !";
        rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/core*
        rm /fred/oz002/users/mmiles/MPTA_GW/partim_frank/pp_8/core*
    else echo "no segfault !";
        ((i++));
    fi;
done


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata
rm -f "job_trackers/ptmcmc341_${1}_${2}_${3}"

echo done