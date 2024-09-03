#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" -a --me | grep milan | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_PMWN.list
    for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/final_checks/ready_for_PMWN_PSRADD10.list); 
    do 
        echo ${psr%.tim}
        sh /home/mmiles/soft/GW/ozstar2/rerun_noise_live400_pbilby_psradd_10_PMWN_check.sh ${psr%.tim};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_PMWN.list
    rm /fred/oz002/users/mmiles/MPTA_GW/partim_frank/psradd_10/core*
    rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/core*
    sleep 10m    
done
