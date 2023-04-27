#!/bin/bash

# hopefully this reruns the cross correlations

psr1=$1
psr2=$2
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | wc -l)

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/cross_corrs/live_1000/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_live_1000_CrossCorr" == $(grep -w -m 1 ^${psr1}_${psr2}_live_1000_CrossCorr /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | wc -l) -lt 1001 ]]; then
    sbatch -J ${psr1}_${psr2}_live_1000_CrossCorr /home/mmiles/soft/GW/ozstar2/cross_corr_mpi_slurm.sh ${psr1} ${psr2} 1000
    echo "rerunning ${psr1}_${psr2}_live_1000_CrossCorr" >> /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/cross_corrs/live_500/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_live_500_CrossCorr" == $(grep -w -m 1 ^${psr1}_${psr2}_live_500_CrossCorr /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | wc -l) -lt 1001 ]]; then
    sbatch -J ${psr1}_${psr2}_live_500_CrossCorr /home/mmiles/soft/GW/ozstar2/cross_corr_mpi_slurm.sh ${psr1} ${psr2} 500
    echo "rerunning ${psr1}_${psr2}_live_500_CrossCorr" >> /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list
fi


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/cross_corrs/fixed_amp_500/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_fixed_amp_live_500_CrossCorr" == $(grep -w -m 1 ^${psr1}_${psr2}_fixed_amp_live_500_CrossCorr /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | wc -l) -lt 1001 ]]; then
    sbatch -J ${psr1}_${psr2}_fixed_amp_live_500_CrossCorr /home/mmiles/soft/GW/ozstar2/cross_corr_fixedAMP_mpi_slurm.sh ${psr1} ${psr2} 500
    echo "rerunning ${psr1}_${psr2}_fixed_amp_live_500_CrossCorr" >> /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/cross_corrs/fixed_amp_ER_500/${psr1}_${psr2}/${psr1}_${psr2}_result.json" ]] && [[ ! "${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr" == $(grep -w -m 1 ^${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list | wc -l) -lt 1001 ]]; then
    sbatch -J ${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr /home/mmiles/soft/GW/ozstar2/cross_corr_fixedAMP_ER_mpi_slurm.sh ${psr1} ${psr2} 500
    echo "rerunning ${psr1}_${psr2}_fixed_amp_live_ER_500_CrossCorr" >> /fred/oz002/users/mmiles/MPTA_GW/cross_corr_slurm.list
fi