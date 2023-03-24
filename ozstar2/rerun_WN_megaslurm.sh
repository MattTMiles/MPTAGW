#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models (PMWN AND PMWN + SW)

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/WN/${psr}/${psr}_WN/WN_result.json" ]] && [[ ! "${psr}_WN" == $(cat /fred/oz002/users/mmiles/MPTA_GW/WN_ozstar2_slurm.list | grep -w ^${psr}_WN) ]]; then
    sbatch -J ${psr}_WN /home/mmiles/soft/GW/ozstar2/WN_noise_mpi_slurm.sh ${psr} WN "wn_tester" 200
    echo "rerunning ${psr}_WN"
fi

