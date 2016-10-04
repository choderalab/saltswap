from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import integrators
from openmmtools.testsystems import WaterBox

import sys
sys.path.append("/Users/rossg/Work/saltswap/saltswap")

size = 25.0*unit.angstrom     # The length of the edges of the water box.
temperature = 300*unit.kelvin
pressure = 1*unit.atmospheres
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=9*unit.angstrom,ewaldErrorTolerance=1E-6)

compound = True
if compound == True:
    integrator = openmm.CompoundIntegrator()
    integrator.addIntegrator(openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, 2.0*unit.femtoseconds))
    integrator.addIntegrator(integrators.VelocityVerletIntegrator(timestep = 1*unit.femtoseconds))
    vv = integrator.getIntegrator(1)
else:
    integrator = integrators.VelocityVerletIntegrator(timestep = 1*unit.femtoseconds)
    vv = integrator


ctype = 'CPU'
if ctype == 'CUDA':
    platform = openmm.Platform.getPlatformByName(ctype)
    properties = {'CudaPrecision': 'mixed'}
    context = openmm.Context(wbox.system, integrator, platform, properties)
else:
    context = openmm.Context(wbox.system, integrator)
context.setPositions(wbox.positions)
context.setVelocitiesToTemperature(temperature)

if compound == True:
    integrator.setCurrentIntegrator(0)
    integrator.step(10)
    integrator.setCurrentIntegrator(1)
else:
    vv.step(10)

ntrials = 10
force = wbox.system.getForce(2)       # Non-bonded force.

# Turning the first water molecule into Na+. The second water molecule into Cl-
run = False
if run == True:
    vv.step(1)     # propagation
    for n in range(ntrials):
        # Get the energy BEFORE the parameters are perturbed.
        state = context.getState(getEnergy=True)
        potential_old = state.getPotentialEnergy()/unit.kilojoule_per_mole
        print('old',potential_old)
         # Perturbation
        fraction = 1 - float(n + 1)/ntrials

        # Water 1 into cation
        force.setParticleParameters(0,charge=-0.834*fraction + (1-fraction)*1.0,sigma=0.3150752406575124*fraction + (1-fraction)*0.2439281,epsilon=0.635968*fraction + (1-fraction)*0.0874393)
        force.setParticleParameters(1,charge=0.417*fraction + (1-fraction)*0.0 ,sigma = 0.0, epsilon = 0.0)
        force.setParticleParameters(2,charge=0.417*fraction,sigma=0,epsilon= 0.0)

        # Water 2 into anion
        force.setParticleParameters(3,charge=-0.834*fraction + (1-fraction)*(-1.0),sigma=0.3150752406575124*fraction + (1-fraction)*0.4477657,epsilon=0.635968*fraction + (1-fraction)*0.0355910)
        force.setParticleParameters(4,charge=0.417*fraction + (1-fraction)*0.0 ,sigma = 0.0, epsilon = 0.0)
        force.setParticleParameters(5,charge=0.417*fraction,sigma=0,epsilon= 0.0)

        force.updateParametersInContext(context)

        vv.step(1)

        state = context.getState(getEnergy=True)
        potential_new = state.getPotentialEnergy()/unit.kilojoule_per_mole
        print('new',potential_new)
    # Reseting the parameters
    force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124*fraction,epsilon=0.635968*fraction)
    force.setParticleParameters(1,charge=0.417,sigma=0,epsilon=1)
    force.setParticleParameters(2,charge=0.417,sigma=0,epsilon=1)
    force.setParticleParameters(3,charge=-0.834,sigma=0.3150752406575124*fraction,epsilon=0.635968*fraction)
    force.setParticleParameters(4,charge=0.417,sigma=0,epsilon=1)
    force.setParticleParameters(5,charge=0.417,sigma=0,epsilon=1)
    force.updateParametersInContext(context)

# Decoupling the first water molecule
run = True
if run == True:
    vv.step(1)     # propagation
    for n in range(ntrials):
        # Get the energy BEFORE the parameters are perturbed.
        state = context.getState(getEnergy=True)
        potential_old = state.getPotentialEnergy()/unit.kilojoule_per_mole
        print('old',potential_old)
         # Perturbation
        fraction = 1 - float(n + 1)/ntrials

        force.setParticleParameters(0,charge=-0.834*fraction,sigma=0.3150752406575124*fraction,epsilon=0.635968*fraction)
        force.setParticleParameters(1,charge=0.417*fraction,sigma=0,epsilon=1*fraction)
        force.setParticleParameters(2,charge=0.417*fraction,sigma=0,epsilon=1*fraction)
        force.updateParametersInContext(context)

        vv.step(1)
        state = context.getState(getEnergy=True)
        potential_new = state.getPotentialEnergy()/unit.kilojoule_per_mole
        print('new',potential_new)
    force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124*fraction,epsilon=0.635968*fraction)
    force.setParticleParameters(1,charge=0.417,sigma=0,epsilon=1)
    force.setParticleParameters(2,charge=0.417,sigma=0,epsilon=1)
    force.updateParametersInContext(context)