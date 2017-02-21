#!/bin/bash

PLAT='CPU'
LEN=30

# Python 2.7, openmm 7.1
conda env create -f python27_openmm71.yml
source activate py27-minimal-openmm-7.1
python time_update_parameters.py -l $LEN -o py27_openmm71_plat_${PLAT}_len_${LEN}.pstat --platform $PLAT
source deactivate py27-minimal-openmm-7.1

# Python 3.5, openmm 7.1
conda env create -f python35_openmm71.yml
source activate py35-minimal-openmm-7.1
python time_update_parameters.py -l $LEN -o py35_openmm71_plat_${PLAT}_len_${LEN}.pstat --platform $PLAT
source deactivate py35-minimal-openmm-7.1


# Python 2.7, openmm 7.2 dev
conda env create -f python27_openmm72.yml
source activate py27-minimal-openmm-7.2
python time_update_parameters.py -l $LEN -o py27_openmm72_plat_${PLAT}_len_${LEN}.pstat --platform $PLAT
source deactivate py27-minimal-openmm-7.2

# Python 3.5, openmm 7.2 dev
conda env create -f python35_openmm72.yml
source activate py35-minimal-openmm-7.2
python time_update_parameters.py -l $LEN -o py35_openmm72_plat_${PLAT}_len_${LEN}.pstat --platform $PLAT
source deactivate py35-minimal-openmm-7.2
