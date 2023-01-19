#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do    
    for psr in $(cd /fred/oz002/users/mmiles/MPTA_GW/partim_investigations/ && ls *par);  
    #for psr in J1614-2230 J1832-0836;
    do 
        echo ${psr%.par}
        #sleep 2s
        sh /home/mmiles/soft/GW/rerun_noise_altpar_live200_megaslurm.sh ${psr%.par};
        sh /home/mmiles/soft/GW/rerun_altpar_PM_WN_live400_megaslurm.sh ${psr%.par};
    done
    sleep 5m    
done