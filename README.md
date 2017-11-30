[![DOI](https://zenodo.org/badge/79260659.svg)](https://zenodo.org/badge/latestdoi/79260659)
[![experimental](http://badges.github.io/stability-badges/dist/experimental.svg)](http://github.com/badges/stability-badges)
[![Build Status](https://travis-ci.org/choderalab/saltswap.svg?branch=master)](https://travis-ci.org/choderalab/saltswap/branches)


# saltswap

`saltswap` simulates dynamic anion-cation (salt) explicit solvent environments in [OpenMM](http://openmm.org).
`saltswap` uses the semi-grand canonical ensemble to couple a simulation box to a reservoir at a given macroscopic salt concentration.
The applied chemical potential in `saltswap` reflects the salt concentration of the reservoir.

## Citations

Please cite the following:
[![DOI](https://zenodo.org/badge/doi/10.1101/226001.svg)](http://dx.doi.org/10.1101/226001)
```
@article{saltswap,
    author = {Gregory A. Ross, Ari\"{e}n S. Rustenburg, Patrick B. Grinaway, Josh Fass, and John D. Chodera.},
    title = {Biomolecular simulations under realistic macroscopic salt conditions},
    journal = {bioRxiv},
    doi = {10.1101/226001}
    year = {2017},
}
```

## Installation

To install:
```bash
python setup.py install
```
Dependencies are currently listed in `saltswap/devtools/conda-recipe/meta.yaml`.

## Quick example

We'll first create a box water in which to try the insertion and deletion of salt:
```python
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
wbox = WaterBox(box_edge=25.0 * unit.angstrom, nonbondedMethod=app.PME,
        cutoff=10 * unit.angstrom, ewaldErrorTolerance=1E-4)
```
Initializing the sampler that will perform Markov chain Monte Carlo by mixing
molecular dynamics moves that transform pairs of water molecules to anions and cations:
```python
from saltswap.swapper import MCMCSampler
sampler = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem=800)
```                      
The chemical potential is specified by `delta_chem` and is in kJ/mol.
The chemical potential corresponding to each macroscopic salt concentration must be calibrated for each water and ion pair forcefield, as well as the specific nonbonded treatment.

By default, `saltswap` makes instantaneous swaps between water and salt.
To achieve usable acceptance rates, we use [nonequilibrium candidate Monte Carlo (NCMC)](http://dx.doi.org/10.1073/pnas.1106094108).

To run a simulation for 1000 iterations of MD and salt-swap moves, we use the `move` method:
```python
sampler.move(1000)
```
More detailed examples can be found in `examples/`.

## Contributors

* Gregory Ross: <gregory.ross@choderalab.org>
* Bas Rustenburg: <bas.rustenburg@choderalab.org>
* John D. Chodera: <john.chodera@choderalab.org>
