from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from mcmc_samplers import MCMCSampler
import numpy as np
from time import time
import matplotlib.pyplot as plt

# Setup box of water
size = 25.0*unit.angstrom     # The length of the edges of the water box.
temperature = 300*unit.kelvin
pressure = 1*unit.atmospheres
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=9*unit.angstrom,ewaldErrorTolerance=1E-5)

# Set up sampler class
npert = 1000
nprop = 1
steps = 500
delta_chem = -10000   # To insert straight away
attempts = 100

timestep = 1.0*unit.femtoseconds
sampler = MCMCSampler(wbox.system, wbox.topology, wbox.positions, temperature=temperature, pressure=pressure,
                      npert=npert, nprop=nprop, propagator='GHMC', ncmc_timestep = timestep,
                      delta_chem=delta_chem, mdsteps=steps, saltsteps=attempts, platform='OpenCL')

# Thermalize
equilibration = 1000
sampler.gen_config(mdsteps=equilibration)

# Add one molecule of salt
nosalt = True
while nosalt:
    sampler.saltswap.update(sampler.context, nattempts=1, cost=-10000)
    n_wats, n_ions, n_ions = sampler.saltswap.getIdentityCounts()
    nosalt = (n_ions == 0)


# Now attempt to only delete salt but never accept
t0 = time()
sampler.saltswap.update(sampler.context, nattempts=attempts, cost=-10000,saltmax=1)
print('Time =', time() - t0)

import pickle
pickle.dump(sampler.saltswap.work_rm_per_step, open('work_rm.pickle', "wb" ))

# plot
#plt.clf()
#for i in range(len(sampler.saltswap.work_add_per_step)):
#    plt.plot(sampler.saltswap.work_add_per_step[i],color='blue', alpha=0.3)
#plt.grid()
#plt.xlabel('NCMC step')
#plt.ylabel('Work per step (kT)')
#plt.savefig('Step_Work_Add.png', format="png")