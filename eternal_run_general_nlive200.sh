#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do    
    #for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise/possible_chrom_wide.list); 
    for psr in J1327-0755 J1811-2405 J2145-0750;
    do 
        echo $psr
        #sleep 2s
        sh /home/mmiles/soft/GW/rerun_noise_live200_general_megaslurm.sh ${psr};
    done
    sleep 5m    
done