#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do    
    sh /home/mmiles/soft/GW/rerun_PTAGW_megaslurm.sh;
    sleep 10m    
done