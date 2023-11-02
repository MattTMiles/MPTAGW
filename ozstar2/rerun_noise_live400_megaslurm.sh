#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l)

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM/DM_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM "efac_c equad_c ecorr_c dm"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN/RN_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN "efac_c equad_c ecorr_c red"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM/RN_DM_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM "efac_c equad_c ecorr_c red dm"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_CHROM/RN_CHROM_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROM" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROM ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROM "efac_c equad_c ecorr_c red chrom"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_CHROM/RN_DM_CHROM_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROM "efac_c equad_c ecorr_c red dm chrom"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_CHROM/DM_CHROM_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROM" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROM ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROM "efac_c equad_c ecorr_c dm chrom"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_CHROM/CHROM_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROM" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROM ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROM "efac_c equad_c ecorr_c chrom"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_CHROMCIDX/CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_c chrom_cidx"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_CHROMCIDX/RN_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROMCIDX "efac_c equad_c ecorr_c red chrom_cidx"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_CHROMCIDX/RN_DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_CHROMCIDX/DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_SW/SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} SW "efac_c equad_c ecorr_c sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_SWDET/SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} SWDET "efac_c equad_c ecorr_c swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_SW/DM_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_SW "efac_c equad_c ecorr_c dm sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_SW/RN_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_SW "efac_c equad_c ecorr_c red sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_SW/RN_DM_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_SW "efac_c equad_c ecorr_c red dm sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_CHROM_SW/RN_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROM_SW "efac_c equad_c ecorr_c red chrom sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_CHROM_SW/RN_DM_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROM_SW "efac_c equad_c ecorr_c red dm chrom sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_CHROM_SW/DM_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROM_SW "efac_c equad_c ecorr_c dm chrom sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_CHROM_SW/CHROM_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROM_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROM_SW "efac_c equad_c ecorr_c chrom sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_CHROMCIDX_SW/CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROMCIDX_SW "efac_c equad_c ecorr_c chrom_cidx sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_CHROMCIDX_SW/RN_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROMCIDX_SW "efac_c equad_c ecorr_c red chrom_cidx sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_CHROMCIDX_SW/RN_DM_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROMCIDX_SW "efac_c equad_c ecorr_c red dm chrom_cidx sw"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_CHROMCIDX_SW/DM_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROMCIDX_SW "efac_c equad_c ecorr_c dm chrom_cidx sw" 
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_SWDET/DM_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_SWDET "efac_c equad_c ecorr_c dm swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_SWDET/RN_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_SWDET "efac_c equad_c ecorr_c red swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_SWDET/RN_DM_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_SWDET "efac_c equad_c ecorr_c red dm swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_CHROM_SWDET/RN_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROM_SWDET "efac_c equad_c ecorr_c red chrom swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_CHROM_SWDET/RN_DM_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROM_SWDET "efac_c equad_c ecorr_c red dm chrom swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_CHROM_SWDET/DM_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROM_SWDET "efac_c equad_c ecorr_c dm chrom swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_CHROM_SWDET/CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROM_SWDET "efac_c equad_c ecorr_c chrom swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_CHROMCIDX_SWDET/CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROMCIDX_SWDET "efac_c equad_c ecorr_c chrom_cidx swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_CHROMCIDX_SWDET/RN_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROMCIDX_SWDET "efac_c equad_c ecorr_c red chrom_cidx swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_RN_DM_CHROMCIDX_SWDET/RN_DM_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROMCIDX_SWDET "efac_c equad_c ecorr_c red dm chrom_cidx swdet"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/${psr}_DM_CHROMCIDX_SWDET/DM_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROMCIDX_SWDET "efac_c equad_c ecorr_c dm chrom_cidx swdet" 
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

# inc. runs with a fixed gamma SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_SGWB/DM_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_SGWB "efac_c equad_c ecorr_c dm gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_SGWB/RN_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_SGWB "efac_c equad_c ecorr_c red gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_SGWB/RN_DM_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_SGWB "efac_c equad_c ecorr_c red dm gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_CHROM_SGWB/RN_CHROM_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROM_SGWB "efac_c equad_c ecorr_c red chrom gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_CHROM_SGWB/RN_DM_CHROM_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROM_SGWB "efac_c equad_c ecorr_c red dm chrom gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_CHROM_SGWB/DM_CHROM_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROM_SGWB "efac_c equad_c ecorr_c dm chrom gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_CHROM_SGWB/CHROM_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROM_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROM_SGWB "efac_c equad_c ecorr_c chrom gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_CHROMCIDX_SGWB/CHROMCIDX_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROMCIDX_SGWB "efac_c equad_c ecorr_c chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_CHROMCIDX_SGWB/RN_CHROMCIDX_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROMCIDX_SGWB "efac_c equad_c ecorr_c red chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_CHROMCIDX_SGWB/RN_DM_CHROMCIDX_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROMCIDX_SGWB "efac_c equad_c ecorr_c red dm chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_CHROMCIDX_SGWB/DM_CHROMCIDX_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROMCIDX_SGWB "efac_c equad_c ecorr_c dm chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_SW_SGWB/SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} SW_SGWB "efac_c equad_c ecorr_c sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_SWDET_SGWB/SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} SWDET_SGWB "efac_c equad_c ecorr_c swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_SW_SGWB/DM_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_SW_SGWB "efac_c equad_c ecorr_c dm sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_SW_SGWB/RN_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_SW_SGWB "efac_c equad_c ecorr_c red sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_SW_SGWB/RN_DM_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_SW_SGWB "efac_c equad_c ecorr_c red dm sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_CHROM_SW_SGWB/RN_CHROM_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROM_SW_SGWB "efac_c equad_c ecorr_c red chrom sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_CHROM_SW_SGWB/RN_DM_CHROM_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROM_SW_SGWB "efac_c equad_c ecorr_c red dm chrom sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_CHROM_SW_SGWB/DM_CHROM_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROM_SW_SGWB "efac_c equad_c ecorr_c dm chrom sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_CHROM_SW_SGWB/CHROM_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROM_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROM_SW_SGWB "efac_c equad_c ecorr_c chrom sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_CHROMCIDX_SW_SGWB/CHROMCIDX_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_CHROMCIDX_SW_SGWB/RN_CHROMCIDX_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c red chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_CHROMCIDX_SW_SGWB/RN_DM_CHROMCIDX_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c red dm chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_CHROMCIDX_SW_SGWB/DM_CHROMCIDX_SW_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c dm chrom_cidx sw gw_const_gamma" 
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_SWDET_SGWB/DM_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_SWDET_SGWB "efac_c equad_c ecorr_c dm swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_SWDET_SGWB/RN_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_SWDET_SGWB "efac_c equad_c ecorr_c red swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_SWDET_SGWB/RN_DM_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_SWDET_SGWB "efac_c equad_c ecorr_c red dm swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_CHROM_SWDET_SGWB/RN_CHROM_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROM_SWDET_SGWB "efac_c equad_c ecorr_c red chrom swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_CHROM_SWDET_SGWB/RN_DM_CHROM_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROM_SWDET_SGWB "efac_c equad_c ecorr_c red dm chrom swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_CHROM_SWDET_SGWB/DM_CHROM_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROM_SWDET_SGWB "efac_c equad_c ecorr_c dm chrom swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_CHROM_SWDET_SGWB/CHROM_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROM_SWDET_SGWB "efac_c equad_c ecorr_c chrom swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_CHROMCIDX_SWDET_SGWB/CHROMCIDX_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_CHROMCIDX_SWDET_SGWB/RN_CHROMCIDX_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c red chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_RN_DM_CHROMCIDX_SWDET_SGWB/RN_DM_CHROMCIDX_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} RN_DM_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c red dm chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_RN_DM_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_DM_CHROMCIDX_SWDET_SGWB/DM_CHROMCIDX_SWDET_SGWB_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET_SGWB ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} DM_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c dm chrom_cidx swdet gw_const_gamma" 
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_DM_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ultra/live_400_ozstar2_SMBHB/${psr}/N1/${psr}_SMBHB_CONST_ER/SMBHB_CONST_ER_result.json" ]] && [[ ! "${psr}_live_400_ozstar2_SMBHB_CONST_ER" == $(grep -w -m 1 ^${psr}_live_400_ozstar2_SMBHB_CONST_ER /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]]; then
    sbatch -J ${psr}_live_400_ozstar2_SMBHB_CONST_ER ~/soft/GW/ozstar2/all_noise_live400_mpi_ultranest_slurm.sh ${psr} SMBHB_CONST_ER "efac_c equad_c ecorr_c smbhb_const extra_red" 
    echo "rerunning ${psr}_live_400_ozstar2_SMBHB_CONST_ER" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi
