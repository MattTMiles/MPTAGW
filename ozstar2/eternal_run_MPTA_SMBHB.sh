#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

i=0; 

while [ $i > -1 ]; 
do    
    squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/SMBHB/PTA_CRN_SMBHB_slurm.list
    sh /home/mmiles/soft/GW/ozstar2/rerun_MPTA_SMBHB_CIRC_PTMCMC_megaslurm.sh
    rm /fred/oz002/users/mmiles/MPTA_GW/SMBHB/PTA_CRN_SMBHB_slurm.list
    rm /fred/oz002/users/mmiles/MPTA_GW/partim_frank/pp_8/core*
    rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/core*
    sleep 10m
done