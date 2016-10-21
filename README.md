# openmm-saltswap
** UNDER DEVELOPMENT **
Code to fluctuate the number of anion-cation (salt) pairs in an explicit water simulation given a salt concentration of a bath.

To use SaltSwap in its current form, add `saltswap/` to your `PYTHONPATH`.

## Manifest

`saltswap/`

```
    saltswap.py		        - Python driver for implementing salt-water exchanges in explicit water simulations
```
```
    mcmc_samplers.py		- Contains wrappers for SaltSwap that allow easy combination with molecular dynamics and SAMS.
```
```
    integrators.py		    - Custom integrator(s) for NCMC moves in SaltSwap.
```
`examples/`

```
    run_sampler.py	        - Command line tool for running salt-swap simulations on a pure box of water
```

`tests/`

```
    Directory containing code tests
```

`Acceptancy-Study/`

```
    Investigation into the optimal NCMC protocol. For delevopment purposes only. 
```
`Performance/`

```
    Contains scripts that have investigated the time efficiency of SaltSwap. For delevopment purposes only. 
```
`SAMS/`

```
   ** DEPRICATED ** Old test scripts for self adjusted mixture sampling 
```
## Authors

    * Gregory A. Ross
