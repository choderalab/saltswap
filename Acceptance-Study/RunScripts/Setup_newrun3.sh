#!/bin/bash

nkernal=(512)
nverlet=(126 256 512 1024)

for k in ${nkernal[*]}
do
    for v in ${nverlet[*]}
    do
        mkdir "nk${k}_nV${v}"
        cd "nk${k}_nV${v}"
        sed "s/REPLACE/--npert $k --nprop $v/" ../submit_dummy > submit
        qsub submit -N "nk${k}nV${v}"
	cd ../
    done
done
