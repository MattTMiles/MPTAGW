#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH -o %x.out
#SBATCH --mem-per-cpu=1gb



source ~/.bashrc
#export OMP_NUM_THREADS=1
ml conda

ml gcc/12.2.0
ml openmpi/4.1.4

conda activate mpippcgw
cd /fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/noise_realiser/J1909-3744/
ulimit -s 16384

touch "noise_realiser_${1}"

echo "noise_realiser_${1}"


python /home/mmiles/soft/GW/plotting_scripts/bootstrap_noise_realise.py -pulsar J1909-3744 -directory /fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/noise_realiser/J1909-3744/ -noise red dm sw ecorr -idx ${1}


cd /fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/noise_realiser/J1909-3744/

rm -f "noise_realiser_${1}"

echo done
