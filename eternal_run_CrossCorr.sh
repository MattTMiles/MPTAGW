#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

i=0; 

while [ $i > -1 ]; 
do   
    echo >> /fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_temp.list
    squeue --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' >> /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list
    for psr1 in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise/factorised_likelihood.list); 
    do
        for psr2 in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise/factorised_likelihood.list);
        do 
            
            if [[ ! "${psr1}_${psr2}" == $(grep -w -m 1 ^${psr1}_${psr2} /fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_temp.list) ]] && [[ ! "${psr2}_${psr1}" == $(grep -w -m 1 ^${psr2}_${psr1} /fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_temp.list) ]] && [[ ! "${psr1}" == "${psr2}" ]]; then
                echo $psr1 $psr2

                sh /home/mmiles/soft/GW/rerun_cross_corrs_megaslurm.sh ${psr1} ${psr2};
                echo "${psr1}_${psr2}" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_temp.list
            fi
        done
    done
    rm /fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_temp.list
    rm /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list

    sleep 10m    
done 
