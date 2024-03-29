#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list
    #for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt); 
    #for psr in J1643-1224 J1721-2457 J1802-2124 J1804-2717;
    for psr in J1721-2457 J1802-2124 J1804-2717;
    do 
        echo $psr
        #sleep 2s
        sh /home/mmiles/soft/GW/rerun_noise_portrait_tester_live200_megaslurm.sh ${psr};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list
    sleep 10m    
done