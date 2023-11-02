#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_WN_test/WN_test_final_res.json" ]] && [[ ! "${psr}_WN_test" == $(grep -w -m 1 ^${psr}_WN_test /fred/oz002/users/mmiles/MPTA_GW/pbilby_noise_search.list) ]]; then
    sbatch -J ${psr}_WN_test /home/mmiles/soft/GW/ozstar2/pbilby_slurm.sh ${psr} WN_test "efac equad ecorr_gauss dm red" 400
    echo "rerunning ${psr}_WN_test" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_noise_search.list
fi
