#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=white_noise_red_noise
#SBATCH --mem=1gb
#SBATCH --tmp=1gb


source ~/.bashrc


touch "${1}_white_noise_red_noise"

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

srun python /home/mmiles/soft/GW/enterprise_run.py -pulsar $1 -results white_noise_red_noise -noise_search efac equad ecorr dm red -sampler bilby -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/total_params.json -pool 1

rm -f "${1}_white_noise_red_noise"


echo done
