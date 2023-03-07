#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr1=$1
psr2=$2
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_corrs/live_1000/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_live_1000_CrossCorr" == $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | grep -w ^${psr1}_${psr2}_live_1000_CrossCorr | head -1) ]]; then
    sbatch -J ${psr1}_${psr2}_live_1000_CrossCorr /home/mmiles/soft/GW/cross_corr_mpi_slurm.sh ${psr1} ${psr2} 1000
    echo "rerunning ${psr1}_${psr2}_live_1000_CrossCorr"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_corrs/live_500/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_live_500_CrossCorr" == $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | grep -w ^${psr1}_${psr2}_live_500_CrossCorr | head -1) ]]; then
    sbatch -J ${psr1}_${psr2}_live_500_CrossCorr /home/mmiles/soft/GW/cross_corr_mpi_slurm.sh ${psr1} ${psr2} 500
    echo "rerunning ${psr1}_${psr2}_live_500_CrossCorr"
fi


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_corrs/fixed_amp_500/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_fixed_amp_live_500_CrossCorr" == $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | grep -w ^${psr1}_${psr2}_fixed_amp_live_500_CrossCorr | head -1) ]]; then
    sbatch -J ${psr1}_${psr2}_fixed_amp_live_500_CrossCorr /home/mmiles/soft/GW/cross_corr_fixedAMP_mpi_slurm.sh ${psr1} ${psr2} 500
    echo "rerunning ${psr1}_${psr2}_fixed_amp_live_500_CrossCorr"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_corrs/fixed_amp_ER_500/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr" == $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | grep -w ^${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr | head -1) ]]; then
    sbatch -J ${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr /home/mmiles/soft/GW/cross_corr_fixedAMP_ER_mpi_slurm.sh ${psr1} ${psr2} 500
    echo "rerunning ${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr"
fi