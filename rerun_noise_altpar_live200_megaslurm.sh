#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM/DM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM "efac_c equad_c ecorr_c dm"
    echo "rerunning ${psr}_alt_par_live_200_DM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN/RN_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN "efac_c equad_c ecorr_c red"
    echo "rerunning ${psr}_alt_par_live_200_RN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_WN/WN_result.json" ]] && [[ ! "${psr}_alt_par_live_200_WN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_WN) ]]; then
    sbatch -J ${psr}_alt_par_live_200_WN ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} WN "efac equad ecorr"
    echo "rerunning ${psr}_alt_par_live_200_WN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_WN_NO_ECORR/WN_NO_ECORR_result.json" ]] && [[ ! "${psr}_alt_par_live_200_WN_NO_ECORR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_WN_NO_ECORR) ]]; then
    sbatch -J ${psr}_alt_par_live_200_WN_NO_ECORR ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} WN_NO_ECORR "efac equad"
    echo "rerunning ${psr}_alt_par_live_200_WN_NO_ECORR"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RED_DM/RED_DM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RED_DM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RED_DM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RED_DM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RED_DM "efac_c equad_c ecorr_c red dm"
    echo "rerunning ${psr}_alt_par_live_200_RED_DM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RED_CHROM/RED_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RED_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RED_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RED_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RED_CHROM "efac_c equad_c ecorr_c red chrom"
    echo "rerunning ${psr}_alt_par_live_200_RED_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RED_DM_CHROM/RED_DM_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RED_DM_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RED_DM_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RED_DM_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RED_DM_CHROM "efac_c equad_c ecorr_c red dm chrom"
    echo "rerunning ${psr}_alt_par_live_200_RED_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_CHROM/DM_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_CHROM "efac_c equad_c ecorr_c dm chrom"
    echo "rerunning ${psr}_alt_par_live_200_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RED_BL/RED_BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RED_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RED_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RED_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RED_BL "efac_c equad_c ecorr_c red band_low"
    echo "rerunning ${psr}_alt_par_live_200_RED_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RED_BH/RED_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RED_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RED_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RED_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RED_BH "efac_c equad_c ecorr_c red band_high"
    echo "rerunning ${psr}_alt_par_live_200_RED_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL/DM_BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL "efac_c equad_c ecorr_c dm band_low"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BH/DM_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BH "efac_c equad_c ecorr_c dm band_high"
    echo "rerunning ${psr}_alt_par_live_200_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BL_BH/RN_BL_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BL_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BL_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BL_BH "efac_c equad_c ecorr_c red band_low band_high"
    echo "rerunning ${psr}_alt_par_live_200_RN_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL_BH/DM_BL_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL_BH "efac_c equad_c ecorr_c dm band_low band_high"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM/CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM "efac_c equad_c ecorr_c chrom"
    echo "rerunning ${psr}_alt_par_live_200_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM_BL/CHROM_BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM_BL "efac_c equad_c ecorr_c chrom band_low"
    echo "rerunning ${psr}_alt_par_live_200_CHROM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM_BH/CHROM_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM_BH "efac_c equad_c ecorr_c chrom band_high"
    echo "rerunning ${psr}_alt_par_live_200_CHROM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM_BL_BH/CHROM_BL_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM_BL_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM_BL_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM_BL_BH "efac_c equad_c ecorr_c chrom band_low band_high"
    echo "rerunning ${psr}_alt_par_live_200_CHROM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROMCIDX/CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_c chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROMCIDX_BL/CHROMCIDX_BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROMCIDX_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROMCIDX_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROMCIDX_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROMCIDX_BL "efac_c equad_c ecorr_c chrom_cidx band_low"
    echo "rerunning ${psr}_alt_par_live_200_CHROMCIDX_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROMCIDX_BH/CHROMCIDX_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROMCIDX_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROMCIDX_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROMCIDX_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROMCIDX_BH "efac_c equad_c ecorr_c chrom_cidx band_high"
    echo "rerunning ${psr}_alt_par_live_200_CHROMCIDX_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROMCIDX_BL_BH/CHROMCIDX_BL_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROMCIDX_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROMCIDX_BL_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROMCIDX_BL_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROMCIDX_BL_BH "efac_c equad_c ecorr_c chrom_cidx band_low band_high"
    echo "rerunning ${psr}_alt_par_live_200_CHROMCIDX_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL/RN_DM_BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL "efac_c equad_c ecorr_c red dm band_low"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL_BH/RN_DM_BL_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL_BH "efac_c equad_c ecorr_c red dm band_low band_high"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BH/RN_DM_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BH "efac_c equad_c ecorr_c red dm band_high"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BL_CHROM/RN_BL_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BL_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BL_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BL_CHROM "efac_c equad_c ecorr_c red band_low chrom"
    echo "rerunning ${psr}_alt_par_live_200_RN_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BH_CHROM/RN_BH_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BH_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BH_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BH_CHROM "efac_c equad_c ecorr_c red band_high chrom"
    echo "rerunning ${psr}_alt_par_live_200_RN_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL_CHROM/DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL_CHROM "efac_c equad_c ecorr_c dm band_low chrom"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BH_CHROM/DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BH_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BH_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BH_CHROM "efac_c equad_c ecorr_c dm band_high chrom"
    echo "rerunning ${psr}_alt_par_live_200_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BL_BH_CHROM/RN_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BL_BH_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BL_BH_CHROM "efac_c equad_c ecorr_c red band_low band_high chrom"
    echo "rerunning ${psr}_alt_par_live_200_RN_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL_BH_CHROM/DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL_BH_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL_BH_CHROM "efac_c equad_c ecorr_c dm band_low band_high chrom"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL_CHROM/RN_DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL_CHROM "efac_c equad_c ecorr_c red dm band_low chrom"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL_BH_CHROM/RN_DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROM "efac_c equad_c ecorr_c red dm band_low band_high chrom"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BH_CHROM/RN_DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BH_CHROM) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BH_CHROM ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BH_CHROM "efac_c equad_c ecorr_c red dm band_high chrom"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_CHROMCIDX/RN_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_CHROMCIDX "efac_c equad_c ecorr_c red chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_CHROMCIDX/RN_DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BL_CHROMCIDX/RN_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BL_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BL_CHROMCIDX "efac_c equad_c ecorr_c red band_low chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BH_CHROMCIDX/RN_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BH_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_high chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL_CHROMCIDX/DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL_CHROMCIDX "efac_c equad_c ecorr_c dm band_low chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BH_CHROMCIDX/DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BH_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_high chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BL_BH_CHROMCIDX/RN_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BL_BH_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_low band_high chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL_BH_CHROMCIDX/DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL_BH_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL_CHROMCIDX/RN_DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL_BH_CHROMCIDX/RN_DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL_BH_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BH_CHROMCIDX/RN_DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BH_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_high chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_CHROMCIDX/DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_CHROMCIDX) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_CHROMCIDX ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx"
    echo "rerunning ${psr}_alt_par_live_200_DM_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_BH/BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} BH "efac_c equad_c ecorr_c band_high"
    echo "rerunning ${psr}_alt_par_live_200_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_BL/BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} BL "efac_c equad_c ecorr_c band_low"
    echo "rerunning ${psr}_alt_par_live_200_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_BH_BL/BH_BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_BH_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_BH_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_BH_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} BH_BL "efac_c equad_c ecorr_c band_high band_low"
    echo "rerunning ${psr}_alt_par_live_200_BH_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RED_CHROM_WIDE/RED_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RED_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RED_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RED_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RED_CHROM_WIDE "efac_c equad_c ecorr_c red chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RED_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RED_DM_CHROM_WIDE/RED_DM_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RED_DM_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RED_DM_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RED_DM_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RED_DM_CHROM_WIDE "efac_c equad_c ecorr_c red dm chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RED_DM_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_CHROM_WIDE/DM_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_CHROM_WIDE "efac_c equad_c ecorr_c dm chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_DM_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM_WIDE/CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM_WIDE "efac_c equad_c ecorr_c chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM_WIDE_BL/CHROM_WIDE_BL_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM_WIDE_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM_WIDE_BL) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM_WIDE_BL ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM_WIDE_BL "efac_c equad_c ecorr_c chrom_wide band_low"
    echo "rerunning ${psr}_alt_par_live_200_CHROM_WIDE_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM_WIDE_BH/CHROM_WIDE_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM_WIDE_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM_WIDE_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM_WIDE_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM_WIDE_BH "efac_c equad_c ecorr_c chrom_wide band_high"
    echo "rerunning ${psr}_alt_par_live_200_CHROM_WIDE_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_CHROM_WIDE_BL_BH/CHROM_WIDE_BL_BH_result.json" ]] && [[ ! "${psr}_alt_par_live_200_CHROM_WIDE_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_CHROM_WIDE_BL_BH) ]]; then
    sbatch -J ${psr}_alt_par_live_200_CHROM_WIDE_BL_BH ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} CHROM_WIDE_BL_BH "efac_c equad_c ecorr_c chrom_wide band_low band_high"
    echo "rerunning ${psr}_alt_par_live_200_CHROM_WIDE_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BL_CHROM_WIDE/RN_BL_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BL_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BL_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BL_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BL_CHROM_WIDE "efac_c equad_c ecorr_c red band_low chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RN_BL_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BH_CHROM_WIDE/RN_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BH_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BH_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BH_CHROM_WIDE "efac_c equad_c ecorr_c red band_high chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RN_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL_CHROM_WIDE/DM_BL_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL_CHROM_WIDE "efac_c equad_c ecorr_c dm band_low chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BH_CHROM_WIDE/DM_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BH_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BH_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BH_CHROM_WIDE "efac_c equad_c ecorr_c dm band_high chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_DM_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_BL_BH_CHROM_WIDE/RN_BL_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_BL_BH_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_BL_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_BL_BH_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_BL_BH_CHROM_WIDE "efac_c equad_c ecorr_c red band_low band_high chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RN_BL_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_DM_BL_BH_CHROM_WIDE/DM_BL_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_DM_BL_BH_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_DM_BL_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_DM_BL_BH_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} DM_BL_BH_CHROM_WIDE "efac_c equad_c ecorr_c dm band_low band_high chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_DM_BL_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL_CHROM_WIDE/RN_DM_BL_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL_CHROM_WIDE "efac_c equad_c ecorr_c red dm band_low chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BL_BH_CHROM_WIDE/RN_DM_BL_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROM_WIDE "efac_c equad_c ecorr_c red dm band_low band_high chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BL_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_altpar/live_200/${psr}_RN_DM_BH_CHROM_WIDE/RN_DM_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_alt_par_live_200_RN_DM_BH_CHROM_WIDE" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_alt_par_live_200_RN_DM_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_alt_par_live_200_RN_DM_BH_CHROM_WIDE ~/soft/GW/all_noise_alt_par_live200_mpi_slurm.sh ${psr} RN_DM_BH_CHROM_WIDE "efac_c equad_c ecorr_c red dm band_high chrom_wide"
    echo "rerunning ${psr}_alt_par_live_200_RN_DM_BH_CHROM_WIDE"
fi


#cd /fred/oz002/users/mmiles/MPTA_GW/enterprise/rerun_logs/
#print_date=$(date)
#touch "${psr} has been rerun on ${print_date}"