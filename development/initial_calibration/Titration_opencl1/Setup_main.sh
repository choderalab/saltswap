#!/bin/bash

POTENTIAL=(750 760 770 780 790 800 810 820 830 840 850)

for U in ${POTENTIAL[*]}
do
    mkdir "potential_${U}"
    cd "potential_${U}"
    sed "s/REPLACE/-u $U/" ../submit_dummy > submit
    qsub submit -N "u${U}"
    cd ../
done
