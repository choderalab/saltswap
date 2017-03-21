## Rough simulations that explore how the number of salt pairs varies with the chemical potential 

### Aims
1. See if `saltswap` simulations produced reasonable results when salt molecules were titrated in by varying the chemical potential.
2. Test SAMS to for calculate relative hydration free energies of salt and water.

### Results in brief
 * The most stable simulations were produced using the `CUDA` GPU kernal. More work is required to verify this. 
 * Using the `CUDA`, SAMS (with the binuary update scheme) was used to predict the relative hydration free energies.
    * With 3 repeats SAMS predicted $\Delta F = 317.18 +/- 0.32$ (in thermal units at 300K)
    * Using BAR on the NCMC work values from the SAMS simulations predicted $\Delta F = -317.28 +/- 0.21$ (in thermal units at 300K)
    * Previous alchemical simulations (with TI, MBAR to follow) predicted $\Delta F = -316.91 +/- 0.19$ (in thermal units at 300K)
    * (Errors are 95% confidence intervals).