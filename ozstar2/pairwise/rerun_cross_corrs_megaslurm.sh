#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr1=$1
psr2=$2
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise



if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/${psr1}_${psr2}/${psr1}_${psr2}_final_res.json" ]] && [[ ! "${psr1}_${psr2}_PBILBY" == $(grep -w -m 1 ^${psr1}_${psr2}_PBILBY /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/cross_corr_slurm.list) ]]; then
    sbatch --mem-per-cpu=1900MB -J ${psr1}_${psr2}_PBILBY /home/mmiles/soft/GW/ozstar2/pairwise/pbilby_apptainer_crosscorr_slurm.sh ${psr1} ${psr2}
    echo "rerunning ${psr1}_${psr2}_PBILBY" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/cross_corr_slurm.list
    #((counter++))
fi