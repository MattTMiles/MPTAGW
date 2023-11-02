#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list | wc -l)

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM/DM_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM" == $(grep -w -m 1 ^${psr}_PBILBY_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM "efac_c equad_c ecorr_gauss_c dm"
    echo "rerunning ${psr}_PBILBY_DM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN/RN_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN" == $(grep -w -m 1 ^${psr}_PBILBY_RN /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN "efac_c equad_c ecorr_gauss_c red"
    echo "rerunning ${psr}_PBILBY_RN" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM/RN_DM_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM "efac_c equad_c ecorr_gauss_c red dm"
    echo "rerunning ${psr}_PBILBY_RN_DM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROM/RN_CHROM_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROM "efac_c equad_c ecorr_gauss_c red chrom_wide"
    echo "rerunning ${psr}_PBILBY_RN_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROM/RN_DM_CHROM_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROM "efac_c equad_c ecorr_gauss_c red dm chrom_wide"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROM/DM_CHROM_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROM "efac_c equad_c ecorr_gauss_c dm chrom_wide"
    echo "rerunning ${psr}_PBILBY_DM_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROM/CHROM_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROM "efac_c equad_c ecorr_gauss_c chrom_wide"
    echo "rerunning ${psr}_PBILBY_CHROM" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROMCIDX/CHROMCIDX_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_gauss_c chrom_cidx"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROMCIDX/RN_CHROMCIDX_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROMCIDX "efac_c equad_c ecorr_gauss_c red chrom_cidx"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROMCIDX/RN_DM_CHROMCIDX_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_gauss_c red dm chrom_cidx"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROMCIDX/DM_CHROMCIDX_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_gauss_c dm chrom_cidx"
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_SW/SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_SW" == $(grep -w -m 1 ^${psr}_PBILBY_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} SW "efac_c equad_c ecorr_gauss_c sw"
    echo "rerunning ${psr}_PBILBY_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_SWDET/SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} SWDET "efac_c equad_c ecorr_gauss_c swdet"
    echo "rerunning ${psr}_PBILBY_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_SW/DM_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_SW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_SW "efac_c equad_c ecorr_gauss_c dm sw"
    echo "rerunning ${psr}_PBILBY_DM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_SW/RN_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_SW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_SW "efac_c equad_c ecorr_gauss_c red sw"
    echo "rerunning ${psr}_PBILBY_RN_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_SW/RN_DM_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_SW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_SW "efac_c equad_c ecorr_gauss_c red dm sw"
    echo "rerunning ${psr}_PBILBY_RN_DM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROM_SW/RN_CHROM_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_SW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROM_SW "efac_c equad_c ecorr_gauss_c red chrom_wide sw"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROM_SW/RN_DM_CHROM_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_SW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROM_SW "efac_c equad_c ecorr_gauss_c red dm chrom_wide sw"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROM_SW/DM_CHROM_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_SW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROM_SW "efac_c equad_c ecorr_gauss_c dm chrom_wide sw"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROM_SW/CHROM_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_SW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROM_SW "efac_c equad_c ecorr_gauss_c chrom_wide sw"
    echo "rerunning ${psr}_PBILBY_CHROM_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROMCIDX_SW/CHROMCIDX_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROMCIDX_SW "efac_c equad_c ecorr_gauss_c chrom_cidx sw"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROMCIDX_SW/RN_CHROMCIDX_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROMCIDX_SW "efac_c equad_c ecorr_gauss_c red chrom_cidx sw"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROMCIDX_SW/RN_DM_CHROMCIDX_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROMCIDX_SW "efac_c equad_c ecorr_gauss_c red dm chrom_cidx sw"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROMCIDX_SW/DM_CHROMCIDX_SW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_SW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_SW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_SW /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROMCIDX_SW "efac_c equad_c ecorr_gauss_c dm chrom_cidx sw" 
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_SW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_SWDET/DM_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_DM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_SWDET "efac_c equad_c ecorr_gauss_c dm swdet"
    echo "rerunning ${psr}_PBILBY_DM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_SWDET/RN_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_RN_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_SWDET "efac_c equad_c ecorr_gauss_c red swdet"
    echo "rerunning ${psr}_PBILBY_RN_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_SWDET/RN_DM_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_SWDET "efac_c equad_c ecorr_gauss_c red dm swdet"
    echo "rerunning ${psr}_PBILBY_RN_DM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROM_SWDET/RN_CHROM_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROM_SWDET "efac_c equad_c ecorr_gauss_c red chrom_wide swdet"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROM_SWDET/RN_DM_CHROM_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROM_SWDET "efac_c equad_c ecorr_gauss_c red dm chrom_wide swdet"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROM_SWDET/DM_CHROM_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROM_SWDET "efac_c equad_c ecorr_gauss_c dm chrom_wide swdet"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROM_SWDET/CHROM_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROM_SWDET "efac_c equad_c ecorr_gauss_c chrom_wide swdet"
    echo "rerunning ${psr}_PBILBY_CHROM_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROMCIDX_SWDET/CHROMCIDX_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROMCIDX_SWDET "efac_c equad_c ecorr_gauss_c chrom_cidx swdet"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROMCIDX_SWDET/RN_CHROMCIDX_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROMCIDX_SWDET "efac_c equad_c ecorr_gauss_c red chrom_cidx swdet"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROMCIDX_SWDET/RN_DM_CHROMCIDX_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROMCIDX_SWDET "efac_c equad_c ecorr_gauss_c red dm chrom_cidx swdet"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROMCIDX_SWDET/DM_CHROMCIDX_SWDET_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_SWDET" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_SWDET /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_SWDET /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROMCIDX_SWDET "efac_c equad_c ecorr_gauss_c dm chrom_cidx swdet" 
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_SWDET" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

# inc. runs with a fixed gamma SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_SGWB/DM_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_SGWB "efac_c equad_c ecorr_gauss_c dm gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_DM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_SGWB/RN_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_SGWB "efac_c equad_c ecorr_gauss_c red gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_SGWB/RN_DM_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_SGWB "efac_c equad_c ecorr_gauss_c red dm gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROM_SGWB/RN_CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROM_SGWB "efac_c equad_c ecorr_gauss_c red chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROM_SGWB/RN_DM_CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROM_SGWB "efac_c equad_c ecorr_gauss_c red dm chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROM_SGWB/DM_CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROM_SGWB "efac_c equad_c ecorr_gauss_c dm chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROM_SGWB/CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROM_SGWB "efac_c equad_c ecorr_gauss_c chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROMCIDX_SGWB/CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROMCIDX_SGWB "efac_c equad_c ecorr_gauss_c chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROMCIDX_SGWB/RN_CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROMCIDX_SGWB "efac_c equad_c ecorr_gauss_c red chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROMCIDX_SGWB/RN_DM_CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROMCIDX_SGWB "efac_c equad_c ecorr_gauss_c red dm chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROMCIDX_SGWB/DM_CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROMCIDX_SGWB "efac_c equad_c ecorr_gauss_c dm chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_SW_SGWB/SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} SW_SGWB "efac_c equad_c ecorr_gauss_c sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_SWDET_SGWB/SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} SWDET_SGWB "efac_c equad_c ecorr_gauss_c swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_SW_SGWB/DM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_SW_SGWB "efac_c equad_c ecorr_gauss_c dm sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_DM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_SW_SGWB/RN_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_SW_SGWB "efac_c equad_c ecorr_gauss_c red sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_SW_SGWB/RN_DM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_SW_SGWB "efac_c equad_c ecorr_gauss_c red dm sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROM_SW_SGWB/RN_CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROM_SW_SGWB "efac_c equad_c ecorr_gauss_c red chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROM_SW_SGWB/RN_DM_CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROM_SW_SGWB "efac_c equad_c ecorr_gauss_c red dm chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROM_SW_SGWB/DM_CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROM_SW_SGWB "efac_c equad_c ecorr_gauss_c dm chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROM_SW_SGWB/CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROM_SW_SGWB "efac_c equad_c ecorr_gauss_c chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROMCIDX_SW_SGWB/CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_gauss_c chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROMCIDX_SW_SGWB/RN_CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_gauss_c red chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROMCIDX_SW_SGWB/RN_DM_CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_gauss_c red dm chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROMCIDX_SW_SGWB/DM_CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_gauss_c dm chrom_cidx sw gw_const_gamma" 
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_SWDET_SGWB/DM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_SWDET_SGWB "efac_c equad_c ecorr_gauss_c dm swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_DM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_SWDET_SGWB/RN_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_SWDET_SGWB "efac_c equad_c ecorr_gauss_c red swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_SWDET_SGWB/RN_DM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_SWDET_SGWB "efac_c equad_c ecorr_gauss_c red dm swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROM_SWDET_SGWB/RN_CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROM_SWDET_SGWB "efac_c equad_c ecorr_gauss_c red chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROM_SWDET_SGWB/RN_DM_CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROM_SWDET_SGWB "efac_c equad_c ecorr_gauss_c red dm chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROM_SWDET_SGWB/DM_CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROM_SWDET_SGWB "efac_c equad_c ecorr_gauss_c dm chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROM_SWDET_SGWB/CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROM_SWDET_SGWB "efac_c equad_c ecorr_gauss_c chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_CHROMCIDX_SWDET_SGWB/CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_gauss_c chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_CHROMCIDX_SWDET_SGWB/RN_CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_gauss_c red chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_RN_DM_CHROMCIDX_SWDET_SGWB/RN_DM_CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} RN_DM_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_gauss_c red dm chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_DM_CHROMCIDX_SWDET_SGWB/DM_CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} DM_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_gauss_c dm chrom_cidx swdet gw_const_gamma" 
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_linExp/${psr}/${psr}_SMBHB_CONST_ER/SMBHB_CONST_ER_final_res.json" ]] && [[ ! "${psr}_PBILBY_CONST_ER" == $(grep -w -m 1 ^${psr}_PBILBY_CONST_ER /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CONST_ER /home/mmiles/soft/GW/ozstar2/pbilby_slurm_linExp.sh ${psr} SMBHB_CONST_ER "efac_c equad_c ecorr_gauss_c smbhb_const extra_red" 
    echo "rerunning ${psr}_PBILBY_CONST_ER" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi
