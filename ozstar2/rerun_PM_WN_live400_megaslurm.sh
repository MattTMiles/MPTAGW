#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models (PMWN AND PMWN + SW)

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list | wc -l)

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN/${psr}/${psr}_PM_WN/PM_WN_result.json" ]] && [[ ! "${psr}_PM_WN" == $(grep -w -m 1 ^${psr}_PM_WN /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_PM_WN /home/mmiles/soft/GW/ozstar2/PM_WN_noise_mpi_slurm.sh ${psr} PM_WN "PM_WN" 200
    echo "rerunning ${psr}_PM_WN" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN/${psr}/${psr}_PM_WN_SW/PM_WN_SW_result.json" ]] && [[ ! "${psr}_PM_WN_SW" == $(grep -w -m 1 ^${psr}_PM_WN_SW /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list| wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_PM_WN_SW /home/mmiles/soft/GW/ozstar2/PM_WN_noise_mpi_slurm.sh ${psr} PM_WN_SW "PM_WN_SW" 200
    echo "rerunning ${psr}_PM_WN_SW" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN/${psr}/${psr}_WN_SW/WN_SW_result.json" ]] && [[ ! "${psr}_WN_SW" == $(grep -w -m 1 ^${psr}_WN_SW /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_WN_SW /home/mmiles/soft/GW/ozstar2/PM_WN_noise_mpi_slurm.sh ${psr} WN_SW "WN_SW" 200
    echo "rerunning ${psr}_WN_SW" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_ozstar2_slurm.list
    #((counter++))
fi