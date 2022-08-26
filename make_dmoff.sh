#!/bin/sh

psr=$1

dmfile=${psr}_dm.dat
dmofile=${psr}_dmo.dat


parfile=/fred/oz002/users/mmiles/MPTA_GW/pars_with_noise/${psr}.par

cp $parfile tmp.par

tempo2 -output general2 -s '{sat} {tndm} {tndmerr}\n' -outfile ${dmfile} -f tmp.par /fred/oz002/users/mmiles/MPTA_GW/pars_with_noise/${psr}.tim

awk '{print "-dmo", $2}' $dmfile | sed '1i\\' | sed '1i\\' > ${dmofile}

paste ${psr}.tim ${dmofile} > ${psr}_dmo.tim


#!/bin/sh

psr=$1

dmfile=${psr}_dm.dat
dmofile=${psr}_dmo.dat


parfile=/fred/oz002/users/mmiles/MPTA_GW/pars_with_noise/${psr}.par

cp $parfile tmp.par


tempo2 -output general2 -s '{sat} {tndm} {tndmerr}\n' -outfile ${dmfile} -f tmp.par /fred/oz002/users/mmiles/MPTA_GW/pars_with_noise/${psr}.tim

awk '{print "-dmo", $2}' $dmfile | sed '1i\\' | sed '1i\\' > ${dmofile}

paste ${psr}.tim ${dmofile} > ${psr}_dmo.tim


