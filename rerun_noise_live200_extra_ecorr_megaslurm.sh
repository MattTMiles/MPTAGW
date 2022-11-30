#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM/DM_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM "efac_c equad_c ecorr_c dm"
    echo "rerunning ${psr}_extra_ecorr_DM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN/RN_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN "efac_c equad_c ecorr_c red"
    echo "rerunning ${psr}_extra_ecorr_RN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_WN/WN_result.json" ]] && [[ ! "${psr}_extra_ecorr_WN" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_WN) ]]; then
    sbatch -J ${psr}_extra_ecorr_WN ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} WN "efac equad ecorr"
    echo "rerunning ${psr}_extra_ecorr_WN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_WN_NO_ECORR/WN_NO_ECORR_result.json" ]] && [[ ! "${psr}_extra_ecorr_WN_NO_ECORR" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_WN_NO_ECORR) ]]; then
    sbatch -J ${psr}_extra_ecorr_WN_NO_ECORR ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} WN_NO_ECORR "efac equad"
    echo "rerunning ${psr}_extra_ecorr_WN_NO_ECORR"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RED_DM/RED_DM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RED_DM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RED_DM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RED_DM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RED_DM "efac_c equad_c ecorr_c red dm"
    echo "rerunning ${psr}_extra_ecorr_RED_DM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RED_CHROM/RED_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RED_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RED_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RED_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RED_CHROM "efac_c equad_c ecorr_c red chrom"
    echo "rerunning ${psr}_extra_ecorr_RED_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RED_DM_CHROM/RED_DM_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RED_DM_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RED_DM_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RED_DM_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RED_DM_CHROM "efac_c equad_c ecorr_c red dm chrom"
    echo "rerunning ${psr}_extra_ecorr_RED_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_CHROM/DM_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_CHROM "efac_c equad_c ecorr_c dm chrom"
    echo "rerunning ${psr}_extra_ecorr_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RED_BL/RED_BL_result.json" ]] && [[ ! "${psr}_extra_ecorr_RED_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RED_BL) ]]; then
    sbatch -J ${psr}_extra_ecorr_RED_BL ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RED_BL "efac_c equad_c ecorr_c red band_low"
    echo "rerunning ${psr}_extra_ecorr_RED_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RED_BH/RED_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_RED_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RED_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_RED_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RED_BH "efac_c equad_c ecorr_c red band_high"
    echo "rerunning ${psr}_extra_ecorr_RED_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BL/DM_BL_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BL) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BL ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BL "efac_c equad_c ecorr_c dm band_low"
    echo "rerunning ${psr}_extra_ecorr_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BH/DM_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BH "efac_c equad_c ecorr_c dm band_high"
    echo "rerunning ${psr}_extra_ecorr_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_BL_BH/RN_BL_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_BL_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_BL_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_BL_BH "efac_c equad_c ecorr_c red band_low band_high"
    echo "rerunning ${psr}_extra_ecorr_RN_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BL_BH/DM_BL_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BL_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BL_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BL_BH "efac_c equad_c ecorr_c dm band_low band_high"
    echo "rerunning ${psr}_extra_ecorr_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROM/CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROM "efac_c equad_c ecorr_c chrom"
    echo "rerunning ${psr}_extra_ecorr_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROM_BL/CHROM_BL_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROM_BL) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROM_BL ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROM_BL "efac_c equad_c ecorr_c chrom band_low"
    echo "rerunning ${psr}_extra_ecorr_CHROM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROM_BH/CHROM_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROM_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROM_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROM_BH "efac_c equad_c ecorr_c chrom band_high"
    echo "rerunning ${psr}_extra_ecorr_CHROM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROM_BL_BH/CHROM_BL_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROM_BL_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROM_BL_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROM_BL_BH "efac_c equad_c ecorr_c chrom band_low band_high"
    echo "rerunning ${psr}_extra_ecorr_CHROM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROMCIDX/CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_c chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROMCIDX_BL/CHROMCIDX_BL_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROMCIDX_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROMCIDX_BL) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROMCIDX_BL ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROMCIDX_BL "efac_c equad_c ecorr_c chrom_cidx band_low"
    echo "rerunning ${psr}_extra_ecorr_CHROMCIDX_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROMCIDX_BH/CHROMCIDX_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROMCIDX_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROMCIDX_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROMCIDX_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROMCIDX_BH "efac_c equad_c ecorr_c chrom_cidx band_high"
    echo "rerunning ${psr}_extra_ecorr_CHROMCIDX_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_CHROMCIDX_BL_BH/CHROMCIDX_BL_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_CHROMCIDX_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_CHROMCIDX_BL_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_CHROMCIDX_BL_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} CHROMCIDX_BL_BH "efac_c equad_c ecorr_c chrom_cidx band_low band_high"
    echo "rerunning ${psr}_extra_ecorr_CHROMCIDX_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BL/RN_DM_BL_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BL) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BL ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BL "efac_c equad_c ecorr_c red dm band_low"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BL_BH/RN_DM_BL_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BL_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BL_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BL_BH "efac_c equad_c ecorr_c red dm band_low band_high"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BH/RN_DM_BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BH "efac_c equad_c ecorr_c red dm band_high"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_BL_CHROM/RN_BL_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_BL_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_BL_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_BL_CHROM "efac_c equad_c ecorr_c red band_low chrom"
    echo "rerunning ${psr}_extra_ecorr_RN_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_BH_CHROM/RN_BH_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_BH_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_BH_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_BH_CHROM "efac_c equad_c ecorr_c red band_high chrom"
    echo "rerunning ${psr}_extra_ecorr_RN_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BL_CHROM/DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BL_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BL_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BL_CHROM "efac_c equad_c ecorr_c dm band_low chrom"
    echo "rerunning ${psr}_extra_ecorr_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BH_CHROM/DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BH_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BH_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BH_CHROM "efac_c equad_c ecorr_c dm band_high chrom"
    echo "rerunning ${psr}_extra_ecorr_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_BL_BH_CHROM/RN_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_BL_BH_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_BL_BH_CHROM "efac_c equad_c ecorr_c red band_low band_high chrom"
    echo "rerunning ${psr}_extra_ecorr_RN_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BL_BH_CHROM/DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BL_BH_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BL_BH_CHROM "efac_c equad_c ecorr_c dm band_low band_high chrom"
    echo "rerunning ${psr}_extra_ecorr_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BL_CHROM/RN_DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BL_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BL_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BL_CHROM "efac_c equad_c ecorr_c red dm band_low chrom"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BL_BH_CHROM/RN_DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BL_BH_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROM "efac_c equad_c ecorr_c red dm band_low band_high chrom"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BH_CHROM/RN_DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BH_CHROM) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BH_CHROM ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BH_CHROM "efac_c equad_c ecorr_c red dm band_high chrom"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_CHROMCIDX/RN_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_CHROMCIDX "efac_c equad_c ecorr_c red chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_CHROMCIDX/RN_DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_BL_CHROMCIDX/RN_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_BL_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_BL_CHROMCIDX "efac_c equad_c ecorr_c red band_low chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_BH_CHROMCIDX/RN_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_BH_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_high chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BL_CHROMCIDX/DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BL_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BL_CHROMCIDX "efac_c equad_c ecorr_c dm band_low chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BH_CHROMCIDX/DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BH_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_high chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_BL_BH_CHROMCIDX/RN_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_BL_BH_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_low band_high chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_BL_BH_CHROMCIDX/DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_BL_BH_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BL_CHROMCIDX/RN_DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BL_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BL_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BL_BH_CHROMCIDX/RN_DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BL_BH_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_RN_DM_BH_CHROMCIDX/RN_DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_RN_DM_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_RN_DM_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_RN_DM_BH_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} RN_DM_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_high chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_RN_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_DM_CHROMCIDX/DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_extra_ecorr_DM_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_DM_CHROMCIDX) ]]; then
    sbatch -J ${psr}_extra_ecorr_DM_CHROMCIDX ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx"
    echo "rerunning ${psr}_extra_ecorr_DM_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_BH/BH_result.json" ]] && [[ ! "${psr}_extra_ecorr_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_BH) ]]; then
    sbatch -J ${psr}_extra_ecorr_BH ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} BH "efac_c equad_c ecorr_c band_high"
    echo "rerunning ${psr}_extra_ecorr_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_BL/BL_result.json" ]] && [[ ! "${psr}_extra_ecorr_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_BL) ]]; then
    sbatch -J ${psr}_extra_ecorr_BL ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} BL "efac_c equad_c ecorr_c band_low"
    echo "rerunning ${psr}_extra_ecorr_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr/${psr}_BH_BL/BH_BL_result.json" ]] && [[ ! "${psr}_extra_ecorr_BH_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_extra_ecorr_BH_BL) ]]; then
    sbatch -J ${psr}_extra_ecorr_BH_BL ~/soft/GW/all_noise_extraecorr__live200_mpi_slurm.sh ${psr} BH_BL "efac_c equad_c ecorr_c band_high band_low"
    echo "rerunning ${psr}_extra_ecorr_BH_BL"
fi

#cd /fred/oz002/users/mmiles/MPTA_GW/enterprise/rerun_logs/
#print_date=$(date)
#touch "${psr} has been rerun on ${print_date}"