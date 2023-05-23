#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

i=0; 

while [ $i > -1 ]; 
do  
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" -a --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/PTA_ozstar2_slurm.list

    sh /home/mmiles/soft/GW/ozstar2/rerun_PTA_CRN_megaslurm.sh;
    
    rm /fred/oz002/users/mmiles/MPTA_GW/PTA_ozstar2_slurm.list
    rm /fred/oz002/users/mmiles/MPTA_GW/partim/core*
    sleep 15m    
done