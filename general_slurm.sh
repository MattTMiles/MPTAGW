#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=quick_data_reduce
#SBATCH --mem=5gb
#SBATCH --tmp=1gb

source ~/.bashrc

#module load psrchive/1e36de3a8
#module load gcc/9.2.0
#module load tempo2/18e1bf6-gcc-9.2.0-python-3.7.4
module load psrchive/f387fd299-python-3.6.4
conda activate portraits

touch "${1}_data_reducing"

echo ${1}
cd /fred/oz002/users/mmiles/MPTA_GW/new_port_NBs

cp /fred/oz005/timing_processed/PTA/${1}/*/*/*/decimated/J*928chTS*.ar .

/fred/oz005/users/mkeith/dlyfix/dlyfix *ar -e dly

#pam -E /fred/oz002/users/mmiles/MPTA_GW/partim/${1}.par ${1}*dly

psradd -E /fred/oz002/users/mmiles/MPTA_GW/partim/${1}.par -T -o ${1}_grand.dly ${1}*dly

rm ${1}*ar
rm ${1}*zap.dly

rm -f "${1}_data_reducing"

echo done
