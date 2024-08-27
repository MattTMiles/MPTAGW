#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs


counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/PTA_HD_DJR_slurm.list | wc -l)

for i in {1..1000}; do echo $i;


    if [[ ! "MPTA_HD_DJR_PL_HYPER_$i" == $(grep -w -m 1 ^MPTA_HD_DJR_PL_HYPER_$i /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/PTA_HD_DJR_slurm.list)  ]] && [[ counter -le 9999 ]]; then
        sbatch -J MPTA_HD_DJR_PL_HYPER_$i /home/mmiles/soft/GW/ozstar2/PTA_HD_DJR_HYPER_apptainer_slurm.sh $i
        echo "MPTA_HD_DJR_PL_HYPER_$i" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/PTA_HD_DJR_slurm.list
        ((counter++))
    fi

done