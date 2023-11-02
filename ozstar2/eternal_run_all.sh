#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" -a --me | grep milan | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    for psr in $(cd /fred/oz002/users/mmiles/MPTA_GW/SMBHB/active_partim/ && ls *tim); 
    do 
        echo ${psr%.tim}
        sh /home/mmiles/soft/GW/ozstar2/rerun_noise_live400_megaslurm.sh ${psr%.tim};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    rm /fred/oz002/users/mmiles/MPTA_GW/SMBHB/active_partim/core*
    sleep 10m    
done