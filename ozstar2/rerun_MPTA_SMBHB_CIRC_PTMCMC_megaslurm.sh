#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/


counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/SMBHB/PTA_CRN_slurm.list | wc -l)


for i in {1..1000}; do echo $i;

    if [[ ! "MPTA_SMBHB_CIRC_HYPER_$i" == $(grep -w -m 1 ^MPTA_SMBHB_CIRC_HYPER_$i /fred/oz002/users/mmiles/MPTA_GW/SMBHB/PTA_CRN_SMBHB_slurm.list)  ]] && [[ counter -le 9999 ]]; then
        sbatch -J MPTA_SMBHB_CIRC_HYPER_$i /home/mmiles/soft/GW/ozstar2/PTMCMC_SMBHB_CIRC_HYPERMODEL_slurm.sh $i
        echo "MPTA_SMBHB_CIRC_HYPER_$i" >> /fred/oz002/users/mmiles/MPTA_GW/SMBHB/PTA_CRN_SMBHB_slurm.list
        ((counter++))
    fi

    # if [[ ! "MPTA_SMBHB_CIRC_CRN_ER_HYPER_$i" == $(grep -w -m 1 ^MPTA_SMBHB_CIRC_CRN_ER_HYPER_$i /fred/oz002/users/mmiles/MPTA_GW/SMBHB/PTA_CRN_SMBHB_slurm.list)  ]] && [[ counter -le 9999 ]]; then
    #     sbatch -J MPTA_SMBHB_CIRC_CRN_ER_HYPER_$i /home/mmiles/soft/GW/ozstar2/PTMCMC_SMBHB_CIRC_CRN_ER_HYPERMODEL_slurm.sh $i
    #     echo "MPTA_SMBHB_CIRC_CRN_ER_HYPER_$i" >> /fred/oz002/users/mmiles/MPTA_GW/SMBHB/PTA_CRN_SMBHB_slurm.list
    #     ((counter++))
    # fi


done