import numpy as np
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from openmmtools.integrators import ExternalPerturbationLangevinIntegrator
print('OpenMM version: ', openmm.version.full_version)

# Creating a waterbox
wbox = WaterBox(box_edge=20.0*unit.angstrom , nonbondedMethod=app.PME, cutoff=9*unit.angstrom, ewaldErrorTolerance=1E-5)
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

niterations = 20
internal_work = np.zeros(niterations)
external_work = np.zeros(niterations)

# A model of NCMC without perturbation but using updateParametersInContext
for i in range(niterations):
    integrator.setGlobalVariableByName('first_step',0)
    integrator.setGlobalVariableByName('protocol_work',0)
    for s in range(10):
        integrator.step(1)
        initial_external_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
        ###---- Perturbation would go here, but updating parameters anyway----###
        non_bonded_force.updateParametersInContext(context)
        final_external_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
        integrator.step(1)
    print('external work =',final_external_energy - initial_external_energy, 'internal work =', integrator.getGlobalVariableByName('protocol_work'))
