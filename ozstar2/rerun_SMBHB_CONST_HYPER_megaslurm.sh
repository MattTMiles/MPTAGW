#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata



for i in {1..10}; do echo $i;
    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/${psr}/SMBHB_WN_${i}/finished" ]] && [[ ! "${psr}_SMBHB_WN_${i}" == $(grep -w -m 1 ^${psr}_SMBHB_WN_${i} /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_WN_slurm.list)  ]]; then
        sbatch -J ${psr}_SMBHB_WN_${i} /home/mmiles/soft/GW/ozstar2/SMBHB_WN_HYPER_apptainer_slurm.sh ${psr} SMBHB_WN_${i} "smbhb_wn" 
        echo "${psr}_SMBHB_WN_${i}" >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_WN_slurm.list
    fi
done