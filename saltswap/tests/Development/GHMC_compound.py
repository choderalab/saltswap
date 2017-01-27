from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import integrators
from openmmtools.testsystems import WaterBox

# Creating a box if water
size = 20.0*unit.angstrom     # The length of the edges of the water box.
temperature = 300*unit.kelvin
pressure = 1*unit.atmospheres
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=9*unit.angstrom,ewaldErrorTolerance=1E-6)

#      Creating the compound integrator
ghmc = integrators.GHMCIntegrator(temperature, 1/unit.picosecond, 0.1*unit.femtoseconds)
langevin = openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)

integrator = openmm.CompoundIntegrator()
integrator.addIntegrator(langevin)
integrator.addIntegrator(ghmc)
integrator.setCurrentIntegrator(1)

context = openmm.Context(wbox.system, integrator)
context.setPositions(wbox.positions)
context.setVelocitiesToTemperature(temperature)

force = wbox.system.getForce(2)       # Non-bonded force.

#      Getting the energy BEFORE the parameters are perturbed
ghmc.step(1)
state = context.getState(getEnergy=True)
potential_old = state.getPotentialEnergy()/unit.kilojoule_per_mole
print('old',potential_old,ghmc.getGlobalVariableByName('potential_new'))

#      Perturbation. Reducing LJ parameters by 10%
force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124*0.9,epsilon=0.635968*0.9)
force.updateParametersInContext(context)

#      Getting the energy AFTER the parameters have been perturbed
state = context.getState(getEnergy=True)
potential_new = state.getPotentialEnergy()/unit.kilojoule_per_mole
ghmc.step(1)     # The 'old' energy for one step should be the energy immediately after the perturbation
print('new',potential_new,ghmc.getGlobalVariableByName('potential_old'))

#      Resetting the parameters
force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124,epsilon=0.635968)
force.updateParametersInContext(context)


