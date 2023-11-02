#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" -a --me | grep milan | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
    for psr in $(cd /fred/oz002/users/mmiles/MPTA_GW/partim_august23/ && ls *tim); 
    do 
        echo ${psr%.tim}
        sh /home/mmiles/soft/GW/ozstar2/rerun_noise_pbilby_SMBHB_WN.sh ${psr%.tim};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
    rm /fred/oz002/users/mmiles/MPTA_GW/partim_august23/core*
    rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/core*
    sleep 10m    
done