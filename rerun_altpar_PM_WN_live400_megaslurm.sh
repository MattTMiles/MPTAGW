#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/PM_WN/${psr}_PM_WN_ALTPAR/PM_WN_ALTPAR_result.json" ]] && [[ ! "${psr}_PM_WN_ALTPAR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_PM_WN_ALTPAR) ]]; then
    sbatch -J ${psr}_PM_WN_ALTPAR /home/mmiles/soft/GW/PM_WN_altpar_noise_mpi_slurm.sh ${psr} PM_WN_ALTPAR "PM_WN_ALTPAR" 200
    echo "rerunning ${psr}_PM_WN_ALTPAR"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/PM_WN/${psr}_PM_WN_NO_EQUAD_ALTPAR/PM_WN_NO_EQUAD_ALTPAR_result.json" ]] && [[ ! "${psr}_PM_WN_NO_EQUAD_ALTPAR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_PM_WN_NO_EQUAD_ALTPAR) ]]; then
    sbatch -J ${psr}_PM_WN_NO_EQUAD_ALTPAR /home/mmiles/soft/GW/PM_WN_altpar_noise_mpi_slurm.sh ${psr} PM_WN_NO_EQUAD_ALTPAR "PM_WN_NO_EQUAD_ALTPAR" 200
    echo "rerunning ${psr}_PM_WN_NO_EQUAD_ALTPAR"
fi

<< com
if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/PM_WN/${psr}_PM_WN_SW/PM_WN_SW_result.json" ]] && [[ ! "${psr}_PM_WN_SW_ALTPAR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_PM_WN_SW_ALTPAR) ]]; then
    sbatch -J ${psr}_PM_WN_SW /home/mmiles/soft/GW/PM_WN_altpar_noise_mpi_slurm.sh ${psr} PM_WN_SW "PM_WN_SW" 200
    echo "rerunning ${psr}_PM_WN_SW_ALTPAR"
fi
com
