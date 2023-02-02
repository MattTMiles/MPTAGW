#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_400/${psr}_PM_WN/PM_WN_result.json" ]] && [[ ! "${psr}_PM_WN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_PM_WN) ]]; then
    sbatch -J ${psr}_PM_WN /home/mmiles/soft/GW/PM_WN_noise_mpi_slurm.sh ${psr} PM_WN "PM_WN" 400
    echo "rerunning ${psr}_PM_WN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_400/${psr}_PM_WN_SW/PM_WN_SW_result.json" ]] && [[ ! "${psr}_PM_WN_SW" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_PM_WN_SW) ]]; then
    sbatch -J ${psr}_PM_WN_SW /home/mmiles/soft/GW/PM_WN_noise_mpi_slurm.sh ${psr} PM_WN_SW "PM_WN_SW" 400
    echo "rerunning ${psr}_PM_WN_SW"
fi