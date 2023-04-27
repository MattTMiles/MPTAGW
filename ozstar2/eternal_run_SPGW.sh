#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" -a --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/SPGW_ozstar2_slurm.list
    for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list); 
    do 
        echo $psr
        #sleep 2s
        sh /home/mmiles/soft/GW/ozstar2/rerun_SPGW_megaslurm.sh ${psr};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/SPGW_ozstar2_slurm.list
    sleep 1m    
done