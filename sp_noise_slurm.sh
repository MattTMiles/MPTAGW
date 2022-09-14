#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --job-name=white_noise_red_noise_no_ecorr
#SBATCH --mem=1gb
#SBATCH --tmp=1gb


source ~/.bashrc


touch "${1}_white_noise_red_noise_no_ecorr"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

srun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results white_noise_red_noise -noise_search efac equad dm red -sampler bilby -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/total_params.json

rm -f "${1}_white_noise_red_noise_no_ecorr"


echo done
