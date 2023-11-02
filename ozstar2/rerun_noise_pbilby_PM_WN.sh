#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/best_bets


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/best_bets/out_pbilby/${psr}/${psr}_PM_WN/PM_WN_final_res.json" ]] && [[ ! "${psr}_PM_WN" == $(grep -w -m 1 ^${psr}_PM_WN /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_PM_WN /home/mmiles/soft/GW/ozstar2/pbilby_pmwn_slurm.sh ${psr} PM_WN "smbhb_wn" 400
    echo "rerunning ${psr}_PM_WN" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi
