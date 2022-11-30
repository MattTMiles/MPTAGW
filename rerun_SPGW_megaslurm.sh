#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_200/${psr}_SPGW/SPGW_result.json" ]] && [[ ! "${psr}_live_200_SPGW" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_SPGW) ]]; then
    sbatch -J ${psr}_live_200_SPGW ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW "efac_c equad_c ecorr_c spgw"  200 DE438
    echo "rerunning ${psr}_live_200_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_200/${psr}_SPGWC/SPGWC_result.json" ]] && [[ ! "${psr}_live_200_SPGWC" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_SPGWC) ]]; then
    sbatch -J ${psr}_live_200_SPGWC ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC "efac_c equad_c ecorr_c spgwc" 200 DE438
    echo "rerunning ${psr}_live_200_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_400/${psr}_SPGW/SPGW_result.json" ]] && [[ ! "${psr}_live_400_SPGW" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_400_SPGW) ]]; then
    sbatch -J ${psr}_live_400_SPGW ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW "efac_c equad_c ecorr_c spgw" 400 DE438
    echo "rerunning ${psr}_live_400_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_400/${psr}_SPGWC/SPGWC_result.json" ]] && [[ ! "${psr}_live_400_SPGWC" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_400_SPGWC) ]]; then
    sbatch -J ${psr}_live_400_SPGWC ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC "efac_c equad_c ecorr_c spgwc" 400 DE438
    echo "rerunning ${psr}_live_400_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_400/${psr}_SPGWC_DE440/SPGWC_DE440_result.json" ]] && [[ ! "${psr}_live_400_SPGWC_DE440" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_400_SPGWC_DE440) ]]; then
    sbatch -J ${psr}_live_400_SPGWC_DE440 ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC_DE440 "efac_c equad_c ecorr_c spgwc" 400 DE440
    echo "rerunning ${psr}_live_400_SPGWC_DE440"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_400/${psr}_SPGWC_WN/SPGWC_WN_result.json" ]] && [[ ! "${psr}_live_400_SPGWC_WN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_400_SPGWC_WN) ]]; then
    sbatch -J ${psr}_live_400_SPGWC_WN ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC_WN "efac equad ecorr_check spgwc" 400 DE438
    echo "rerunning ${psr}_live_400_SPGWC_WN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_400/${psr}_SPGWC_18_WN/SPGWC_18_WN_result.json" ]] && [[ ! "${psr}_live_400_SPGWC_18_WN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_400_SPGWC_18_WN) ]]; then
    sbatch -J ${psr}_live_400_SPGWC_18_WN ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC_18_WN "efac equad ecorr_check spgwc_18" 400 DE438
    echo "rerunning ${psr}_live_400_SPGWC_18_WN"
fi

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise/rerun_logs/
print_date=$(date)
touch "${psr} single pulsar GW live_400 has been rerun on ${print_date}"