#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

i=0; 

while [ $i > -1 ]; 
do    
    squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_CONST_slurm.list
    #for psr in $(cd /fred/oz002/users/mmiles/MPTA_GW/SMBHB/active_partim/ && ls *tim); 
    #do 
    echo J0613-0200
    sh /home/mmiles/soft/GW/ozstar2/rerun_SMBHB_CONST_HYPER_megaslurm.sh J0613-0200

    #done

    rm /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_CONST_slurm.list
    rm /fred/oz002/users/mmiles/MPTA_GW/SMBHB/active_partim/core*
    sleep 10m
done
