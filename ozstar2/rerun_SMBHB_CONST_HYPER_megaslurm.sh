#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata
req_mem=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/psr_memory_fixedWN.txt | grep ${psr} | awk '{print $2}')


for i in {1..4}; do echo $i;
    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/${psr}/SMBHB_${i}/finished" ]] && [[ ! "${psr}_SMBHB_${i}" == $(grep -w -m 1 ^${psr}_SMBHB_${i} /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_WN_slurm.list)  ]]; then
        sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_SMBHB_${i} /home/mmiles/soft/GW/ozstar2/SMBHB_WN_HYPER_apptainer_slurm.sh ${psr} SMBHB_${i} "efac_c equad_c ecorr_c smbhb_frank_pp8" 
        echo "${psr}_SMBHB_${i}" >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_WN_slurm.list
    fi

    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/${psr}/SMBHB_ER_${i}/finished" ]] && [[ ! "${psr}_SMBHB_ER_${i}" == $(grep -w -m 1 ^${psr}_SMBHB_ER_${i} /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_WN_slurm.list)  ]]; then
        sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_SMBHB_ER_${i} /home/mmiles/soft/GW/ozstar2/SMBHB_WN_HYPER_apptainer_slurm.sh ${psr} SMBHB_ER_${i} "efac_c equad_c ecorr_c smbhb_frank_pp8 extra_red" 
        echo "${psr}_SMBHB_ER_${i}" >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_SMBHB_WN_slurm.list
    fi
done


