#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr1=$1
psr2=$2
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/cross_corr_slurm.list | wc -l)

for i in {1..10}; do echo $i;

    if [[ ! "${psr1}_${psr2}_PTMCMC433_$i" == $(grep -w -m 1 ^${psr1}_${psr2}_PTMCMC433_$i /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/cross_corr_slurm.list) ]] && [[ counter -le 9999 ]] && [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_433/${psr1}_${psr2}/${psr1}_${psr2}_${i}/finished" ]]; then
        sbatch --mem-per-cpu=900MB -J ${psr1}_${psr2}_PTMCMC433_$i /home/mmiles/soft/GW/ozstar2/pairwise/ptmcmc433_apptainer_crosscorr_slurm.sh ${psr1} ${psr2} $i
        echo "rerunning ${psr1}_${psr2}_PTMCMC433_$i" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/cross_corr_slurm.list
        ((counter++))
    fi

    if [[ ! "${psr1}_${psr2}_PTMCMC341_$i" == $(grep -w -m 1 ^${psr1}_${psr2}_PTMCMC341_$i /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/cross_corr_slurm.list) ]] && [[ counter -le 9999 ]] && [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/${psr1}_${psr2}/${psr1}_${psr2}_${i}/finished" ]]; then
        sbatch --mem-per-cpu=900MB -J ${psr1}_${psr2}_PTMCMC341_$i /home/mmiles/soft/GW/ozstar2/pairwise/ptmcmc341_apptainer_crosscorr_slurm.sh ${psr1} ${psr2} $i
        echo "rerunning ${psr1}_${psr2}_PTMCMC341_$i" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/cross_corr_slurm.list
        ((counter++))
    fi
done