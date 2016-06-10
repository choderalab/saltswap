# openmm-saltswap
Code to fluctuate the number of anion-cation (salt) pairs in an explicit water simulation given a salt concentration of a bath.

## Manifest

`saltswap/`

```
    saltswap.py		- Python module for implementing salt-water exchanges in explicit water simulations
```

`examples/`

```
    run_saltswap.py	- Command line tool for running salt-swap simulations given an input PDB
    waterbox.pdb	- PDB file of a box of tip3p water
```

`roughcode/`

```
    Directory were new code is tested
```

`Analysis/Performance/`

```
    Directory that contains a set of tests and analyses for erratic performance on GPUs. 
    Tests revealed code is fine.
```
## Authors

    * Gregory A. Ross (code based on openmm-constph)
