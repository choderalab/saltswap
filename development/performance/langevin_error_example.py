import numpy as np
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from openmmtools.integrators import ExternalPerturbationLangevinIntegrator
print('OpenMM version: ', openmm.version.full_version)

# Using one CPU thread
import os
os.environ['OPENMM_CPU_THREADS'] = '1'

# Long range method
nonbonded_method = 'CutoffPeriodic'

# Creating a waterbox
wbox = WaterBox(box_edge=21.0*unit.angstrom , nonbondedMethod=getattr(app, nonbonded_method))
wbox.system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, 300*unit.kelvin))

# Extracting the nonbonded force
non_bonded_force = wbox.system.getForce(2)

# The integrator to perform the equilibrium dynamics
integrator = ExternalPerturbationLangevinIntegrator(temperature=300*unit.kelvin, collision_rate=50.0 / unit.picosecond, timestep=1.0 * unit.femtosecond)

# Creating the context
platform = openmm.Platform.getPlatformByName('CPU')
context = openmm.Context(wbox.system, integrator, platform)
context.setPositions(wbox.positions)

# Running some equilibrium dynamics
integrator.step(100)

# The number of NCMC type iterations and NCMC steps per iteration.
niterations = 20
ncmc_steps = 10

internal_work = np.zeros(niterations)
external_work = np.zeros(niterations)

# Whether to call updateParametersInContext. If True, then assertion below will fail.
update_parameters = True

# A model of NCMC without perturbation but using updateParametersInContext
for i in range(niterations):
    #integrator.reset_protocol_work()
    #integrator.setGlobalVariableByName('first_step',0)
    integrator.setGlobalVariableByName('protocol_work',0)
    for s in range(ncmc_steps):
        integrator.step(1)
        initial_external_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
        ###---- Not perturbing the system but updating parameters anyway----###
        if update_parameters:
            non_bonded_force.updateParametersInContext(context)
        final_external_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
        integrator.step(1)
    internal_work[i] = integrator.getGlobalVariableByName('protocol_work')
    external_work[i] = final_external_energy - initial_external_energy
assert np.all(np.abs(internal_work - external_work) < 1E-5)
