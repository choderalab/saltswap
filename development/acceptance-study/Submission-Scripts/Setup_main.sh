#!/bin/bash

npert=(1024 2048 4096)
nprop=(1 2 4)

for k in ${npert[*]}
do
    for v in ${nprop[*]}
    do
        if [ -a "prt${k}_prp${v}/run1.txt" ]
        then
            echo 'Simulatation completed'
        else
            mkdir "prt${k}_prp${v}"
            cd "prt${k}_prp${v}"
            sed "s/REPLACE/--npert $k --nprop $v/" ../submit_dummy > submit
            qsub submit -N "t${k}_p${v}"
	    cd ../
        fi
    done
done
