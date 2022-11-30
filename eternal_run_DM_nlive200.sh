#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do    
    for psr in $(cat /fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt); 
    do 
        echo $psr
        #sleep 2s
        sh /home/mmiles/soft/GW/rerun_DM_megaslurm.sh ${psr};
    done
    sleep 5m    
done