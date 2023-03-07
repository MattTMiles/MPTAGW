#!/bin/bash

# rerun_noise_megaslurm.sh psr
# this is just meant to be used as hoc for whatever extra models need to be run

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/${psr}/${psr}_PM_WN_SW_REMOVED_ECORR/PM_WN_SW_REMOVED_ECORR_result.json" ]] && [[ ! "${psr}_live_200_PM_WN_SW_REMOVED_ECORR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_PM_WN_SW_REMOVED_ECORR) ]]; then
    sbatch -J ${psr}_live_200_PM_WN_SW_REMOVED_ECORR ~/soft/GW/all_noise_live200_removal_mpi_slurm.sh ${psr} PM_WN_SW_REMOVED_ECORR "pm_wn_sw" "ecorr"
    echo "rerunning ${psr}_live_200_PM_WN_SW_REMOVED_ECORR"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/${psr}/${psr}_WN_SW_REMOVED_ECORR/WN_SW_REMOVED_ECORR_result.json" ]] && [[ ! "${psr}_live_200_WN_SW_REMOVED_ECORR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_WN_SW_REMOVED_ECORR) ]]; then
    sbatch -J ${psr}_live_200_WN_SW_REMOVED_ECORR ~/soft/GW/all_noise_live200_removal_mpi_slurm.sh ${psr} WN_SW_REMOVED_ECORR "wn_sw" "ecorr"
    echo "rerunning ${psr}_live_200_WN_SW_REMOVED_ECORR"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/${psr}/${psr}_PM_WN_SW_REMOVED_DM/PM_WN_SW_REMOVED_DM_result.json" ]] && [[ ! "${psr}_live_200_PM_WN_SW_REMOVED_DM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_PM_WN_SW_REMOVED_DM) ]]; then
    sbatch -J ${psr}_live_200_PM_WN_SW_REMOVED_DM ~/soft/GW/all_noise_live200_removal_mpi_slurm.sh ${psr} PM_WN_SW_REMOVED_DM "pm_wn_sw" "dm"
    echo "rerunning ${psr}_live_200_PM_WN_SW_REMOVED_DM"
fi

