#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

ulimit -s 16384

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list | wc -l)

for i in {1..200}; do echo $i;

    if [[ ! "MPTA_HD_REWEIGHT_$i" == $(grep -w -m 1 ^MPTA_HD_REWEIGHT_$i /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list)  ]]; then
        sbatch -J MPTA_HD_REWEIGHT_$i /home/mmiles/soft/GW/ozstar2/PTA_HD_reweighting_mpi_slurm.sh $i
        echo "MPTA_HD_REWEIGHT_$i" >> /fred/oz002/users/mmiles/MPTA_GW/PTA_CRN_slurm.list
        #((counter++))
    fi
done
