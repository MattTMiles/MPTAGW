#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models (PMWN AND PMWN + SW)

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/${psr}/${psr}_PM_WN_SW_extra_fbins/PM_WN_SW_extra_fbins_result.json" ]] && [[ ! "${psr}_PM_WN_SW_extra_fbins" == $(cat /fred/oz002/users/mmiles/MPTA_GW/PM_WN_SW_extra_fbins_slurm.list | grep -w ^${psr}_PM_WN_SW_extra_fbins) ]]; then
    sbatch -J ${psr}_PM_WN_SW_extra_fbins /home/mmiles/soft/GW/PM_WN_extra_fbins_eternal_run.sh ${psr} PM_WN_SW_extra_fbins "PM_WN_SW_extra_fbins" 200
    echo "rerunning ${psr}_PM_WN_SW_extra_fbins"
fi

