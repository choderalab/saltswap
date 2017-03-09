#!/bin/bash

# The number of alchemical intermediates
NSTATES=20
# One less than the total number of states, as python indexing starts at 0
MAX_INDEX=$(expr $NSTATES - 1)

for i in $(seq 0 $MAX_INDEX); do
    mkdir state_${i}
    cd state_${i}
    sed "s/REPLACE/--state $i --nstates $NSTATES/" ../submit_dummy > submit
    qsub submit -N "alchem_${i}"
    cd ../
done
