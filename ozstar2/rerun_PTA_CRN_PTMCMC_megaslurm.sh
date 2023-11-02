#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

ulimit -s 16384

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list | wc -l)


for i in {1..1000}; do echo $i;

    if [[ ! "MPTA_CRN_VARYGAMMA_PL_ER_HYPER_$i" == $(grep -w -m 1 ^MPTA_CRN_VARYGAMMA_PL_ER_HYPER_$i /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list)  ]]; then
        sbatch -J MPTA_CRN_VARYGAMMA_PL_ER_HYPER_$i /home/mmiles/soft/GW/ozstar2/PTA_CRN_PL_ER_HYPERMODEL_PTMCMC_mpi_slurm.sh $i
        echo "MPTA_CRN_VARYGAMMA_PL_ER_HYPER_$i" >> /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list
        #((counter++))
    fi


    # if [[ ! "MPTA_CRN_HYPER_FREESPEC_ER_RUN_PTMCMC_$i" == $(grep -w -m 1 ^MPTA_CRN_HYPER_FREESPEC_ER_RUN_PTMCMC_$i /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list)  ]]; then
    #     sbatch -J MPTA_CRN_HYPER_FREESPEC_ER_RUN_PTMCMC_$i /home/mmiles/soft/GW/ozstar2/PTA_CRN_HYPER_FREESPEC_ER_PTMCMC_mpi_slurm.sh $i
    #     echo "MPTA_CRN_HYPER_FREESPEC_ER_RUN_PTMCMC_$i" >> /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list
    #     #((counter++))
    # fi

    # if [[ ! "MPTA_CRN_HYPER_MISSPEC_RUN_PTMCMC_$i" == $(grep -w -m 1 ^MPTA_CRN_HYPER_MISSPEC_RUN_PTMCMC_$i /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list)  ]]; then
    #     sbatch -J MPTA_CRN_HYPER_MISSPEC_RUN_PTMCMC_$i /home/mmiles/soft/GW/ozstar2/PTA_CRN_PL_MISSPEC_HYPERMODEL_PTMCMC_mpi_slurm.sh $i
    #     echo "MPTA_CRN_HYPER_MISSPEC_RUN_PTMCMC_$i" >> /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list
    #     #((counter++))
    # fi

done