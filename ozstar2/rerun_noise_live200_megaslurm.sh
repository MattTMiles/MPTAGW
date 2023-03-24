#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l)

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM/DM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM "efac_c equad_c ecorr_c dm"
    echo "rerunning ${psr}_live_200_ozstar2_DM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN/RN_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN "efac_c equad_c ecorr_c red"
    echo "rerunning ${psr}_live_200_ozstar2_RN" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM/RN_DM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM "efac_c equad_c ecorr_c red dm"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_CHROM/RN_CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_CHROM "efac_c equad_c ecorr_c red chrom"
    echo "rerunning ${psr}_live_200_ozstar2_RN_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_CHROM/RN_DM_CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_CHROM "efac_c equad_c ecorr_c red dm chrom"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_CHROM/DM_CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROM "efac_c equad_c ecorr_c dm chrom"
    echo "rerunning ${psr}_live_200_ozstar2_DM_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROM/CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROM "efac_c equad_c ecorr_c chrom"
    echo "rerunning ${psr}_live_200_ozstar2_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROMCIDX/CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_c chrom_cidx"
    echo "rerunning ${psr}_live_200_ozstar2_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_CHROMCIDX/RN_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_CHROMCIDX "efac_c equad_c ecorr_c red chrom_cidx"
    echo "rerunning ${psr}_live_200_ozstar2_RN_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_CHROMCIDX/RN_DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_CHROMCIDX/DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx"
    echo "rerunning ${psr}_live_200_ozstar2_DM_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_SW/SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} SW "efac_c equad_c ecorr_c sw"
    echo "rerunning ${psr}_live_200_ozstar2_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_SWDET/SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} SWDET "efac_c equad_c ecorr_c swdet"
    echo "rerunning ${psr}_live_200_ozstar2_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_HFRED/HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} HFRED "efac_c equad_c ecorr_c hfred"
    echo "rerunning ${psr}_live_200_ozstar2_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

# Add in SW and SWDET


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_SW/DM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_SW "efac_c equad_c ecorr_c dm sw"
    echo "rerunning ${psr}_live_200_ozstar2_DM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_SW/RN_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_SW "efac_c equad_c ecorr_c red sw"
    echo "rerunning ${psr}_live_200_ozstar2_RN_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_SW/RN_DM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_SW "efac_c equad_c ecorr_c red dm sw"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_CHROM_SW/RN_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_CHROM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_CHROM_SW "efac_c equad_c ecorr_c red chrom sw"
    echo "rerunning ${psr}_live_200_ozstar2_RN_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_CHROM_SW/RN_DM_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_CHROM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_CHROM_SW "efac_c equad_c ecorr_c red dm chrom sw"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_CHROM_SW/DM_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_CHROM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROM_SW "efac_c equad_c ecorr_c dm chrom sw"
    echo "rerunning ${psr}_live_200_ozstar2_DM_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROM_SW/CHROM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROM_SW "efac_c equad_c ecorr_c chrom sw"
    echo "rerunning ${psr}_live_200_ozstar2_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROMCIDX_SW/CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROMCIDX_SW "efac_c equad_c ecorr_c chrom_cidx sw"
    echo "rerunning ${psr}_live_200_ozstar2_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_CHROMCIDX_SW/RN_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_CHROMCIDX_SW "efac_c equad_c ecorr_c red chrom_cidx sw"
    echo "rerunning ${psr}_live_200_ozstar2_RN_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_CHROMCIDX_SW/RN_DM_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX_SW "efac_c equad_c ecorr_c red dm chrom_cidx sw"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_CHROMCIDX_SW/DM_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROMCIDX_SW "efac_c equad_c ecorr_c dm chrom_cidx sw" 
    echo "rerunning ${psr}_live_200_ozstar2_DM_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_HFRED_SW/HFRED_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_HFRED_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_HFRED_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_HFRED_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} HFRED_SW "efac_c equad_c ecorr_c hfred sw"
    echo "rerunning ${psr}_live_200_ozstar2_HFRED_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi



if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_SWDET/DM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_SWDET "efac_c equad_c ecorr_c dm swdet"
    echo "rerunning ${psr}_live_200_ozstar2_DM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_SWDET/RN_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_SWDET "efac_c equad_c ecorr_c red swdet"
    echo "rerunning ${psr}_live_200_ozstar2_RN_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_SWDET/RN_DM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_SWDET "efac_c equad_c ecorr_c red dm swdet"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_CHROM_SWDET/RN_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_CHROM_SWDET "efac_c equad_c ecorr_c red chrom swdet"
    echo "rerunning ${psr}_live_200_ozstar2_RN_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_CHROM_SWDET/RN_DM_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_CHROM_SWDET "efac_c equad_c ecorr_c red dm chrom swdet"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_CHROM_SWDET/DM_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROM_SWDET "efac_c equad_c ecorr_c dm chrom swdet"
    echo "rerunning ${psr}_live_200_ozstar2_DM_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROM_SWDET/CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROM_SWDET "efac_c equad_c ecorr_c chrom swdet"
    echo "rerunning ${psr}_live_200_ozstar2_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROMCIDX_SWDET/CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROMCIDX_SWDET "efac_c equad_c ecorr_c chrom_cidx swdet"
    echo "rerunning ${psr}_live_200_ozstar2_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_CHROMCIDX_SWDET/RN_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_CHROMCIDX_SWDET "efac_c equad_c ecorr_c red chrom_cidx swdet"
    echo "rerunning ${psr}_live_200_ozstar2_RN_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_CHROMCIDX_SWDET/RN_DM_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX_SWDET "efac_c equad_c ecorr_c red dm chrom_cidx swdet"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_CHROMCIDX_SWDET/DM_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROMCIDX_SWDET "efac_c equad_c ecorr_c dm chrom_cidx swdet" 
    echo "rerunning ${psr}_live_200_ozstar2_DM_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_HFRED_SWDET/HFRED_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_HFRED_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_HFRED_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_HFRED_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} HFRED_SWDET "efac_c equad_c ecorr_c hfred swdet"
    echo "rerunning ${psr}_live_200_ozstar2_HFRED_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

## Adding in hfred

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED/DM_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED "efac_c equad_c ecorr_c dm hfred"
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED/RN_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED "efac_c equad_c ecorr_c red hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED/RN_DM_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED "efac_c equad_c ecorr_c red dm hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_CHROM/RN_HFRED_CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_CHROM "efac_c equad_c ecorr_c red chrom hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_CHROM/RN_DM_HFRED_CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_CHROM "efac_c equad_c ecorr_c red dm chrom hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_CHROM/DM_HFRED_CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_CHROM "efac_c equad_c ecorr_c dm chrom hfred"
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROM/CHROM_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROM" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROM ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROM "efac_c equad_c ecorr_c chrom hfred"
    echo "rerunning ${psr}_live_200_ozstar2_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROMCIDX/CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_c chrom_cidx hfred"
    echo "rerunning ${psr}_live_200_ozstar2_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_CHROMCIDX/RN_HFRED_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_CHROMCIDX "efac_c equad_c ecorr_c red chrom_cidx hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_CHROMCIDX/RN_DM_HFRED_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_CHROMCIDX/DM_HFRED_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx hfred"
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_SWDET_HFRED/SWDET_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_SWDET_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_SWDET_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_SWDET_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} SWDET_HFRED "efac_c equad_c ecorr_c swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_SWDET_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_SW/DM_HFRED_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_SW "efac_c equad_c ecorr_c dm sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_SW/RN_HFRED_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_SW "efac_c equad_c ecorr_c red sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_SW/RN_DM_HFRED_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_SW "efac_c equad_c ecorr_c red dm sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_CHROM_SW/RN_HFRED_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_CHROM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_CHROM_SW "efac_c equad_c ecorr_c red chrom sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_CHROM_SW/RN_DM_HFRED_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_CHROM_SW "efac_c equad_c ecorr_c red dm chrom sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_CHROM_SW/DM_HFRED_CHROM_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_CHROM_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_CHROM_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_CHROM_SW "efac_c equad_c ecorr_c dm chrom sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROM_SW_HFRED/CHROM_SW_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROM_SW_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROM_SW_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROM_SW_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROM_SW_HFRED "efac_c equad_c ecorr_c chrom sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_CHROM_SW_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROMCIDX_SW_HFRED/CHROMCIDX_SW_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROMCIDX_SW_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROMCIDX_SW_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROMCIDX_SW_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROMCIDX_SW_HFRED "efac_c equad_c ecorr_c chrom_cidx sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_CHROMCIDX_SW_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_CHROMCIDX_SW/RN_HFRED_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_CHROMCIDX_SW "efac_c equad_c ecorr_c red chrom_cidx sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_CHROMCIDX_SW/RN_DM_HFRED_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_CHROMCIDX_SW "efac_c equad_c ecorr_c red dm chrom_cidx sw hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_CHROMCIDX_SW/DM_HFRED_CHROMCIDX_SW_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SW ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_CHROMCIDX_SW "efac_c equad_c ecorr_c dm chrom_cidx sw hfred" 
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_SWDET/DM_HFRED_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_SWDET "efac_c equad_c ecorr_c dm swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_SWDET/RN_HFRED_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_SWDET "efac_c equad_c ecorr_c red swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_SWDET/RN_DM_HFRED_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_SWDET "efac_c equad_c ecorr_c red dm swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_CHROM_SWDET/RN_HFRED_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_CHROM_SWDET "efac_c equad_c ecorr_c red chrom swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_CHROM_SWDET/RN_DM_HFRED_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_CHROM_SWDET "efac_c equad_c ecorr_c red dm chrom swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_CHROM_SWDET/DM_HFRED_CHROM_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_CHROM_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_CHROM_SWDET "efac_c equad_c ecorr_c dm chrom swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROM_SWDET_HFRED/CHROM_SWDET_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROM_SWDET_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROM_SWDET_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROM_SWDET_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROM_SWDET_HFRED "efac_c equad_c ecorr_c chrom swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_CHROM_SWDET_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_CHROMCIDX_SWDET_HFRED/CHROMCIDX_SWDET_HFRED_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_CHROMCIDX_SWDET_HFRED" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_CHROMCIDX_SWDET_HFRED /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_CHROMCIDX_SWDET_HFRED ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} CHROMCIDX_SWDET_HFRED "efac_c equad_c ecorr_c chrom_cidx swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_CHROMCIDX_SWDET_HFRED" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_HFRED_CHROMCIDX_SWDET/RN_HFRED_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_HFRED_CHROMCIDX_SWDET "efac_c equad_c ecorr_c red chrom_cidx swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_HFRED_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_RN_DM_HFRED_CHROMCIDX_SWDET/RN_DM_HFRED_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_HFRED_CHROMCIDX_SWDET "efac_c equad_c ecorr_c red dm chrom_cidx swdet hfred"
    echo "rerunning ${psr}_live_200_ozstar2_RN_DM_HFRED_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_DM_HFRED_CHROMCIDX_SWDET/DM_HFRED_CHROMCIDX_SWDET_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SWDET ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} DM_HFRED_CHROMCIDX_SWDET "efac_c equad_c ecorr_c dm chrom_cidx swdet hfred" 
    echo "rerunning ${psr}_live_200_ozstar2_DM_HFRED_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_ozstar2.list
    #((counter++))
fi


#cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/rerun_logs/
#print_date=$(date)
#touch "${psr} has been rerun on ${print_date}"