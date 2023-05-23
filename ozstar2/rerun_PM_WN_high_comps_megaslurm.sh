#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models (PMWN AND PMWN + SW)

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list | wc -l)

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/${psr}/${psr}_PM_WN_HC/PM_WN_HC_result.json" ]] && [[ ! "${psr}_PM_WN_HC" == $(grep -w -m 1 ^${psr}_PM_WN_HC /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list | wc -l) -gt -1 ]] && [[ $counter -gt -1 ]]; then
    sbatch -J ${psr}_PM_WN_HC /home/mmiles/soft/GW/ozstar2/PM_WN_HC_noise_mpi_slurm.sh ${psr} PM_WN_HC "PM_WN_HC" 200
    echo "rerunning ${psr}_PM_WN_HC" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/${psr}/${psr}_PM_WN_SW_HC/PM_WN_SW_HC_result.json" ]] && [[ ! "${psr}_PM_WN_SW_HC" == $(grep -w -m 1 ^${psr}_PM_WN_SW_HC /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list| wc -l) -gt -1 ]] && [[ $counter -gt -1 ]]; then
    sbatch -J ${psr}_PM_WN_SW_HC /home/mmiles/soft/GW/ozstar2/PM_WN_HC_noise_mpi_slurm.sh ${psr} PM_WN_SW_HC "PM_WN_SW_HC" 200
    echo "rerunning ${psr}_PM_WN_SW_HC" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list
    #((counter++))
fi