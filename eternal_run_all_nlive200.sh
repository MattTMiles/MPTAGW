#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list
    for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt); 
    #for psr in J1614-2230 J1832-0836;
    do 
        echo $psr
        #sleep 2s
        sh /home/mmiles/soft/GW/rerun_noise_live200_megaslurm.sh ${psr};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list
    sleep 10m    
done