#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_WN/WN_result.json" ]] && [[ ! "${psr}_live_200_WN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_WN) ]]; then
    sbatch -J ${psr}_live_200_WN ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} WN "efac equad ecorr red dm"
    echo "rerunning ${psr}_live_200_WN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_WN_NO_ECORR/WN_NO_ECORR_result.json" ]] && [[ ! "${psr}_live_200_WN_NO_ECORR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_WN_NO_ECORR) ]]; then
    sbatch -J ${psr}_live_200_WN_NO_ECORR ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} WN_NO_ECORR "efac equad red dm"
    echo "rerunning ${psr}_live_200_WN_NO_ECORR"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_WN_NO_EQUAD/WN_NO_EQUAD_result.json" ]] && [[ ! "${psr}_live_200_WN_NO_EQUAD" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_WN_NO_EQUAD) ]]; then
    sbatch -J ${psr}_live_200_WN_NO_EQUAD ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} WN_NO_EQUAD "efac ecorr red dm"
    echo "rerunning ${psr}_live_200_WN_NO_EQUAD"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_WN_NO_EQUAD_NO_ECORR/WN_NO_EQUAD_NO_ECORR_result.json" ]] && [[ ! "${psr}_live_200_WN_NO_EQUAD_NO_ECORR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_WN_NO_EQUAD_NO_ECORR) ]]; then
    sbatch -J ${psr}_live_200_WN_NO_EQUAD_NO_ECORR ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} WN_NO_EQUAD_NO_ECORR "efac red dm"
    echo "rerunning ${psr}_live_200_WN_NO_EQUAD_NO_ECORR"
fi