#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=2
#SBATCH --time=06:00:00
#SBATCH -o %x.out
#SBATCH --mem=10gb
#SBATCH --tmp=10gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}"

echo "${1}"

mpirun python /home/mmiles/soft/GW/enterprise_CRN.py -results $1 -noise_search $2 -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/FL_WN_params.json -nlive 200 -psrlist /fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_trusted_noise_281022.txt -alt_dir out_ppc/trusted_noise/

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}"


echo done
