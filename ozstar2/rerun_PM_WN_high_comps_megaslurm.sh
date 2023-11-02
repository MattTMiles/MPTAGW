#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models (PMWN AND PMWN + SW)

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list | wc -l)

# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/${psr}/${psr}_PM_WN_HC/PM_WN_HC_result.json" ]] && [[ ! "${psr}_PM_WN_HC" == $(grep -w -m 1 ^${psr}_PM_WN_HC /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list) ]]; then
#     sbatch -J ${psr}_PM_WN_HC /home/mmiles/soft/GW/ozstar2/PM_WN_HC_noise_mpi_slurm.sh ${psr} PM_WN_HC "PM_WN_HC" 200
#     echo "rerunning ${psr}_PM_WN_HC" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list
#     #((counter++))
# fi

# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/${psr}/${psr}_PM_WN_SW_HC/PM_WN_SW_HC_result.json" ]] && [[ ! "${psr}_PM_WN_SW_HC" == $(grep -w -m 1 ^${psr}_PM_WN_SW_HC /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list) ]]; then
#     sbatch -J ${psr}_PM_WN_SW_HC /home/mmiles/soft/GW/ozstar2/PM_WN_HC_noise_mpi_slurm.sh ${psr} PM_WN_SW_HC "PM_WN_SW_HC" 200
#     echo "rerunning ${psr}_PM_WN_SW_HC" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list
#     #((counter++))
# fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN/${psr}/${psr}_MPTA_PM/MPTA_PM_result.json" ]] && [[ ! "${psr}_MPTA_PM" == $(grep -w -m 1 ^${psr}_MPTA_PM /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list) ]]; then
    sbatch -J ${psr}_MPTA_PM /home/mmiles/soft/GW/ozstar2/PM_WN_noise_mpi_slurm.sh ${psr} MPTA_PM "MPTA_PM" 600
    echo "rerunning ${psr}_MPTA_PM" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list
    #((counter++))
fi

for i in {1..10}; do echo $i;
    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/MPTA_PM/${psr}/MPTA_PM_${i}/finished" ]] && [[ ! "${psr}_HYPER_MPTA_PM" == $(grep -w -m 1 ^${psr}_HYPER_MPTA_PM /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list) ]]; then
        sbatch -J ${psr}_HYPER_MPTA_PM /home/mmiles/soft/GW/ozstar2/PM_WN_noise_HYPER_slurm.sh ${psr} MPTA_PM_${i} "MPTA_PM"
        echo "rerunning ${psr}_HYPER_MPTA_PM" >> /fred/oz002/users/mmiles/MPTA_GW/PM_WN_HC_ozstar2_slurm.list
    fi
done