#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

i=0; 

while [ $i > -1 ]; 
do    
    squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/PTA_HD_DJR_slurm.list
    sh /home/mmiles/soft/GW/ozstar2/rerun_PTA_HD_DJR_PTMCMC_megaslurm.sh
    rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/PTA_HD_DJR_slurm.list
    rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/HD_runs/core*
    sleep 10m
done