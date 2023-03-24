#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" -a --me | grep milan | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list

    for psr in J1713+0747;
    do 
        echo $psr
        #sleep 2s
        sh /home/mmiles/soft/GW/ozstar2/rerun_noise_live200_WN_megaslurm.sh ${psr};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list
    sleep 5m    
done