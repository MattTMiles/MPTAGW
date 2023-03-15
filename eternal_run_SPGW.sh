#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list
    for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise/factorised_likelihood.list); 
    do 
        echo $psr
        
        sh /home/mmiles/soft/GW/rerun_SPGW_megaslurm.sh ${psr};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list
    sleep 10m    
done