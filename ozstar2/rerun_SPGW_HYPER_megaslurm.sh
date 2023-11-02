#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2


for i in {1..10}; do echo $i;
    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/SPGWC/${psr}/SPGWC_${i}/finished" ]] && [[ ! "${psr}_HYPER_SPGW_${i}" == $(grep -w -m 1 ^${psr}_HYPER_SPGW_${i} /fred/oz002/users/mmiles/MPTA_GW/HYPER_SPGW_slurm.list)  ]]; then
        sbatch -J ${psr}_HYPER_SPGW_${i} /home/mmiles/soft/GW/ozstar2/SPGWC_HYPER_slurm.sh ${psr} SPGWC_${i} "efac_c equad_c ecorr_c spgwc"
        echo "${psr}_HYPER_SPGW_${i}" >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_SPGW_slurm.list

    fi
done

for i in {1..10}; do echo $i;
    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/SPGWC/${psr}/SPGWC_WIDE_${i}/finished" ]] && [[ ! "${psr}_HYPER_SPGW_WIDE_${i}" == $(grep -w -m 1 ^${psr}_HYPER_SPGW_WIDE_${i} /fred/oz002/users/mmiles/MPTA_GW/HYPER_SPGW_slurm.list)  ]]; then
        sbatch -J ${psr}_HYPER_SPGW_WIDE_${i} /home/mmiles/soft/GW/ozstar2/SPGWC_HYPER_slurm.sh ${psr} SPGWC_WIDE_${i} "efac_c equad_c ecorr_c spgwc_wide"
        echo "${psr}_HYPER_SPGW_WIDE_${i}" >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_SPGW_slurm.list

    fi
done

for i in {1..10}; do echo $i;
    if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/SPGWC/${psr}/SPGWC_ER_${i}/finished" ]] && [[ ! "${psr}_HYPER_SPGW_ER_${i}" == $(grep -w -m 1 ^${psr}_HYPER_SPGW_ER_${i} /fred/oz002/users/mmiles/MPTA_GW/HYPER_SPGW_slurm.list)  ]]; then
        sbatch -J ${psr}_HYPER_SPGW_ER_${i} /home/mmiles/soft/GW/ozstar2/SPGWC_HYPER_slurm.sh ${psr} SPGWC_ER_${i} "efac_c equad_c ecorr_c extra_red spgwc"
        echo "${psr}_HYPER_SPGW_ER_${i}" >> /fred/oz002/users/mmiles/MPTA_GW/HYPER_SPGW_slurm.list

    fi
done