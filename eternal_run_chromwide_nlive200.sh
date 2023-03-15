#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list
    #for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise/possible_chrom_wide.list); 
    for psr in J1643-1224;
    do 
        echo $psr
        #sleep 2s
        sh /home/mmiles/soft/GW/rerun_noise_live200_chromwide_megaslurm.sh ${psr};
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list
    sleep 5m    
done