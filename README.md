# SaltSwap
Code to fluctuate the number of anion-cation (salt) pairs in `OpenMM` explicit water simulations. 
`saltswap` uses the semi-grand canonical ensemble to couple a simulation box to a 
reservoir at a given salt concentration. The chemical potential that is applied in 
`saltswap` reflects the salt concentration of the reservoir. 

*WARNING: This code is under development.*

## Installation
```
python setup.py install 
```


## Quick example
We'll create a box water in which to try the insertion and deletion of salt. Assuming
the `openmm` modules have been loaded, type

```
from openmmtools.testsystems import WaterBox
wbox = WaterBox(box_edge=25.0 * unit.angstrom, nonbondedMethod=app.PME,
        cutoff=10 * unit.angstrom, ewaldErrorTolerance=1E-5)
```

Initializing the sampler that will perform Markov chain Monte Carlo by mixing
molecular dynamics moves that transform pairs of water molecules to anions and cations.

```
sampler = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem=800)
```                      

The chemical potential is specified by `delta_chem` and is in KJ/mol. As project is ongoing, it has yet to be
worked out which value of the chemical potential corresponds to which concentration of the salt-water reservoir.

By default, instantaneous swaps between water and salt are made. As the acceptance rate for these moves are so 
low, the code supports swap moves with [nonequilibrium candidate Monte Carlo](https://arxiv.org/abs/1105.2278).

To run a simulation for 1000 iterations of MD and salt-swap moves, type

```
sampler.move(1000)
```

More detailed examples can be found in `examples/`.

## Dependencies
* openmm
* numpy

## Contributors ##
* Gregory Ross <gregory.ross@choderalab.org>
* Bas Rustenburg <bas.rustenburg@choderalab.org>
* John D. Chodera <john.chodera@choderalab.org>