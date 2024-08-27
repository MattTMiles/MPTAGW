#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

i=0; 

while [ $i > -1 ]; 
do    
    squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_FREESPEC_slurm.list
    for psr in $(cd /fred/oz002/users/mmiles/MPTA_GW/partim_august23_snr10/partim_updated/ && ls *tim); 
    do 
        echo ${psr}
        sh /home/mmiles/soft/GW/ozstar2/rerun_FREESPEC_HYPER_megaslurm.sh ${psr%.tim}

    done

    rm /fred/oz002/users/mmiles/MPTA_GW/HYPER_FREESPEC_slurm.list
    rm /fred/oz002/users/mmiles/MPTA_GW/partim_august23_snr10/partim_updated/core*
    sleep 10m
done
