#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/trusted_noise/pl_nocorr_fixgam/pl_nocorr_fixgam_result.json" ]] && [[ ! "pl_nocorr_fixgam" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^pl_nocorr_fixgam) ]]; then
    sbatch -J pl_nocorr_fixgam ~/soft/GW/PTAGW_mpi_slurm.sh pl_nocorr_fixgam "pl_nocorr_fixgam"
    echo "rerunning pl_nocorr_fixgam"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/trusted_noise/pl_orf_bins/pl_orf_bins_result.json" ]] && [[ ! "pl_orf_bins" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^pl_orf_bins) ]]; then
    sbatch -J pl_orf_bins ~/soft/GW/PTAGW_mpi_slurm.sh pl_orf_bins "pl_orf_bins"
    echo "rerunning pl_orf_bins"
fi

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise/rerun_logs/
print_date=$(date)
touch "PTAGW has been rerun on ${print_date}"