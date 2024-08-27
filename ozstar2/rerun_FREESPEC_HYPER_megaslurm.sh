#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata



for i in {1..10}; do echo $i;
    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/FREESPEC/${psr}/FREESPEC_${i}/finished" ]] && [[ ! "${psr}_FREESPEC_${i}" == $(grep -w -m 1 ^${psr}_FREESPEC_${i} /fred/oz002/users/mmiles/MPTA_GW/HYPER_FREESPEC_slurm.list)  ]]; then
        sbatch -J ${psr}_FREESPEC_${i} /home/mmiles/soft/GW/ozstar2/SMBHB_FREESPEC_apptainer_slurm.sh ${psr} FREESPEC_${i} "efac_c equad_c ecorr_gauss_c free_spgw" 
        echo "${psr}_FREESPEC_${i}" >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_FREESPEC_slurm.list
    fi
done
