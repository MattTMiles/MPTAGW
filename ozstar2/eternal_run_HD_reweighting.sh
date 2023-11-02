#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

i=0; 

while [ $i > -1 ]; 
do    
    squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list
    sh /home/mmiles/soft/GW/ozstar2/rerun_HD_reweighting.sh
    rm /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list
    sleep 10m
done