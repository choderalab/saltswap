#!/bin/bash

POTENTIAL=(320.0 322.5 325.0) 

for U in ${POTENTIAL[*]}
do
    mkdir "potential_${U}"
    cd "potential_${U}"
    sed "s/REPLACE/-u $U/" ../submit_dummy > submit
    qsub submit -N "u${U}"
    cd ../
done
