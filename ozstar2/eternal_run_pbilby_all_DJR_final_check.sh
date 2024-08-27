#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" -a --me | grep milan | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list
    for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/DJR_noise/psr_check_done.list); 
    do 
        echo ${psr}
        sh /home/mmiles/soft/GW/ozstar2/rerun_noise_live400_pbilby_DJR_noise_final_check.sh ${psr};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list
    rm /fred/oz002/dreardon/mk_gw/pipeline/djr_final_final_final_august/core*
    rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/core*
    sleep 10m    
done

