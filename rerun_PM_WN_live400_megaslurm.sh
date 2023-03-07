#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models (PMWN AND PMWN + SW)

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/${psr}/${psr}_PM_WN/PM_WN_result.json" ]] && [[ ! "${psr}_PM_WN" == $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_slurm.list | grep -w ^${psr}_PM_WN) ]]; then
    sbatch -J ${psr}_PM_WN /home/mmiles/soft/GW/PM_WN_noise_mpi_slurm.sh ${psr} PM_WN "PM_WN" 200
    echo "rerunning ${psr}_PM_WN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/${psr}/${psr}_PM_WN_SW/PM_WN_SW_result.json" ]] && [[ ! "${psr}_PM_WN_SW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_slurm.list | grep -w ^${psr}_PM_WN_SW) ]]; then
    sbatch -J ${psr}_PM_WN_SW /home/mmiles/soft/GW/PM_WN_noise_mpi_slurm.sh ${psr} PM_WN_SW "PM_WN_SW" 200
    echo "rerunning ${psr}_PM_WN_SW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/${psr}/${psr}_WN_SW/WN_SW_result.json" ]] && [[ ! "${psr}_WN_SW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_slurm.list| grep -w ^${psr}_WN_SW) ]]; then
    sbatch -J ${psr}_WN_SW /home/mmiles/soft/GW/PM_WN_noise_mpi_slurm.sh ${psr} WN_SW "WN_SW" 200
    echo "rerunning ${psr}_WN_SW"
fi