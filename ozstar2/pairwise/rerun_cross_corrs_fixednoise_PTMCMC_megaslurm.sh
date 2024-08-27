#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr1=$1
psr2=$2
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/cross_corr_slurm.list | wc -l)


if [[ ! "${psr1}_${psr2}_corrAmp" == $(grep -w -m 1 ^${psr1}_${psr2}_corrAmp /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/cross_corr_slurm.list) ]] && [[ counter -le 9999 ]] && [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/corrAmp_runs/${psr1}_${psr2}/finished" ]]; then
    sbatch --mem-per-cpu=900MB -J ${psr1}_${psr2}_corrAmp /home/mmiles/soft/GW/ozstar2/pairwise/ptmcmcFIXEDNOISE_corrAmp_apptainer_crosscorr_slurm.sh ${psr1} ${psr2} 
    echo "rerunning ${psr1}_${psr2}_corrAmp" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/cross_corr_slurm.list
    ((counter++))
fi

if [[ ! "${psr1}_${psr2}_corr" == $(grep -w -m 1 ^${psr1}_${psr2}_corr /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/cross_corr_slurm.list) ]] && [[ counter -le 9999 ]] && [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/corr_runs/${psr1}_${psr2}/finished" ]]; then
    sbatch --mem-per-cpu=900MB -J ${psr1}_${psr2}_corr /home/mmiles/soft/GW/ozstar2/pairwise/ptmcmcFIXEDNOISE_corr_apptainer_crosscorr_slurm.sh ${psr1} ${psr2} 
    echo "rerunning ${psr1}_${psr2}_corr" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/cross_corr_slurm.list
    ((counter++))
