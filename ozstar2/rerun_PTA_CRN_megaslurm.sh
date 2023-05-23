#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/PTA_ozstar2_slurm.list | wc -l)


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/SPGW/PTA_RUN/CRN_run_result.json" ]] && [[ ! "MPTA_CRN_VARYGAMMA_PL_RUN" == $(grep -w -m 1 ^MPTA_CRN_VARYGAMMA_PL_RUN /fred/oz002/users/mmiles/MPTA_GW/PTA_ozstar2_slurm.list)  ]]; then
    sbatch -J MPTA_CRN_VARYGAMMA_PL_RUN /home/mmiles/soft/GW/ozstar2/PTA_CRN_PL_mpi_slurm.sh
    echo "MPTA_CRN_VARYGAMMA_PL_RUN" >> /fred/oz002/users/mmiles/MPTA_GW/PTA_ozstar2_slurm.list
    #((counter++))
fi



