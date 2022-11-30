#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_BH/BH_result.json" ]] && [[ ! "${psr}_live_200_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_BH) ]]; then
    sbatch -J ${psr}_live_200_BH ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} BH "efac_c equad_c ecorr_c band_high"
    echo "rerunning ${psr}_live_200_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_BL/BL_result.json" ]] && [[ ! "${psr}_live_200_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_BL) ]]; then
    sbatch -J ${psr}_live_200_BL ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} BL "efac_c equad_c ecorr_c band_low"
    echo "rerunning ${psr}_live_200_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_BH_BL/BH_BL_result.json" ]] && [[ ! "${psr}_live_200_BH_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_BH_BL) ]]; then
    sbatch -J ${psr}_live_200_BH_BL ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} BH_BL "efac_c equad_c ecorr_c band_high band_low"
    echo "rerunning ${psr}_live_200_BH_BL"
fi

#cd /fred/oz002/users/mmiles/MPTA_GW/enterprise/rerun_logs/
#print_date=$(date)
#touch "${psr} has been rerun on ${print_date}"