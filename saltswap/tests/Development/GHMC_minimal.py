from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import integrators
from openmmtools.testsystems import WaterBox

import sys
sys.path.append("/Users/rossg/Work/saltswap/saltswap")
from integrators import GHMCIntegrator as GHMC

size = 20.0*unit.angstrom     # The length of the edges of the water box.
temperature = 300*unit.kelvin
pressure = 1*unit.atmospheres
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=9*unit.angstrom,ewaldErrorTolerance=1E-6)
#ghmc = integrators.GHMCIntegrator(temperature, 1/unit.picosecond, 0.1*unit.femtoseconds)
#ghmc = GHMC(temperature, 1/unit.picosecond, 0.1*unit.femtoseconds)    # the modified integrator

compound = True
if compound == True:
    integrator = openmm.CompoundIntegrator()
    #integrator.addIntegrator(integrators.GHMCIntegrator(temperature, 1.0/unit.picosecond, 0.1*unit.femtoseconds))
    integrator.addIntegrator(openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, 2.0*unit.femtoseconds))
    integrator.addIntegrator(integrators.GHMCIntegrator(temperature, 1.0/unit.picosecond, 0.1*unit.femtoseconds))
    ghmc = integrator.getIntegrator(1)
else:
    integrator = integrators.GHMCIntegrator(temperature, 1/unit.picosecond, 0.1*unit.femtoseconds)
    ghmc = integrator

context = openmm.Context(wbox.system, integrator)
context.setPositions(wbox.positions)
context.setVelocitiesToTemperature(temperature)

if compound == True:
    integrator.setCurrentIntegrator(0)
    integrator.step(10)
    integrator.setCurrentIntegrator(1)
else:
    ghmc.step(10)

ntrials = 10
force = wbox.system.getForce(2)       # Non-bonded force.

run = True
if run == True:
    ghmc.step(1)     # propagation
    for n in range(ntrials):
        # Get the energy BEFORE the parameters are perturbed.
        state = context.getState(getEnergy=True)
        potential_old = state.getPotentialEnergy()/unit.kilojoule_per_mole
        print('old',potential_old,ghmc.getGlobalVariableByName('potential_new'))
         # Perturbation
        fraction = 1 - float(n + 1)/ntrials
        #force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124*fraction,epsilon=0.635968*fraction)
        force.setParticleParameters(0,charge=-0.834*fraction,sigma=0.3150752406575124*fraction,epsilon=0.635968*fraction)
        force.setParticleParameters(1,charge=0.417*fraction,sigma=0,epsilon=1*fraction)
        force.setParticleParameters(2,charge=0.417*fraction,sigma=0,epsilon=1*fraction)
        force.updateParametersInContext(context)
        # Get the energy AFTER the parameters have been perturbed.
        state = context.getState(getEnergy=True)
        potential_new = state.getPotentialEnergy()/unit.kilojoule_per_mole
        ghmc.step(1)     # The 'old' energy for one step should be the energy immediately after the perturbation
        print('new',potential_new,ghmc.getGlobalVariableByName('potential_old'))
    #force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124,epsilon=0.635968)
    force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124*fraction,epsilon=0.635968*fraction)
    force.setParticleParameters(1,charge=0.417,sigma=0,epsilon=1)
    force.setParticleParameters(2,charge=0.417,sigma=0,epsilon=1)
    force.updateParametersInContext(context)



print('\nNEW:\n')

o = 0
n = 0
for step in range(5):
    ghmc.step(1)
    state = context.getState(getEnergy=True)
    potential_old = state.getPotentialEnergy()/unit.kilojoule_per_mole
    print('old',potential_old,ghmc.getGlobalVariableByName('potential_new'))
    if potential_old == ghmc.getGlobalVariableByName('potential_new'): o += 1
    # Perturbation
    force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124*0.9,epsilon=0.635968*0.9)
    force.updateParametersInContext(context)
    # Get the energy AFTER the parameters have been perturbed.
    state = context.getState(getEnergy=True)
    potential_new = state.getPotentialEnergy()/unit.kilojoule_per_mole
    ghmc.step(1)     # The 'old' energy for one step should be the energy immediately after the perturbation
    print('new',potential_new,ghmc.getGlobalVariableByName('potential_old'))
    if potential_new == ghmc.getGlobalVariableByName('potential_old'): n += 1
    force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124,epsilon=0.635968)
    force.updateParametersInContext(context)


print o,n