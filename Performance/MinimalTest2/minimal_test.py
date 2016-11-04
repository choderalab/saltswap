import time
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from simtk.openmm import LangevinIntegrator
from integrators import GHMCIntegrator

platform = openmm.Platform.getPlatformByName('OpenCL')
properties = {'Precision': 'mixed'}
box_edge = 10.0 * unit.angstrom
wbox = WaterBox(box_edge=box_edge, cutoff=box_edge / 2.05, nonbondedMethod=app.PME)
force = wbox.system.getForce(2) # NonbondedForce
nsteps = 50000 # number of switching steps
temperature = 300.0 * unit.kelvin
collision_rate = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
def set_lambda(context, lambda_value=1.0):
    force.setParticleParameters(0,charge=-0.834*lambda_value,sigma=0.3150752406575124*lambda_value,epsilon=0.635968*lambda_value)
    force.setParticleParameters(1,charge=0.417*lambda_value,sigma=0,epsilon=1*lambda_value)
    force.setParticleParameters(2,charge=0.417*lambda_value,sigma=0,epsilon=1*lambda_value)
    force.updateParametersInContext(context)

def create_context(integrator):
    context = openmm.Context(wbox.system, integrator, platform, properties)
    context.setPositions(wbox.positions)
    context.setVelocitiesToTemperature(300*unit.kelvin)
    set_lambda(context, 1.0)
    integrator.step(100)
    return context

print('System has %d particles' % wbox.system.getNumParticles())
format = '%64s: %8.3f s for %8d steps (%8.3f ps) : %8.3f ms / step'

# Time LangevinIntegrator without switching
integrator = LangevinIntegrator(temperature, collision_rate, timestep)
context = create_context(integrator)
initial_time = time.time()
for step in range(nsteps):
    integrator.step(1)
elapsed_time = time.time() - initial_time
del context, integrator
print(format % ('LangevinIntegrator', elapsed_time, nsteps, nsteps*timestep/unit.picoseconds, 1000*elapsed_time/float(nsteps)))

# Time LangevinIntegrator with switching
integrator = LangevinIntegrator(temperature, collision_rate, timestep)
context = create_context(integrator)
initial_time = time.time()
for step in range(nsteps):
    set_lambda(context, lambda_value=float(nsteps-step)/float(nsteps))
    integrator.step(1)
elapsed_time = time.time() - initial_time
del context, integrator
print(format % ('LangevinIntegrator with updateParametersInContext', elapsed_time, nsteps, nsteps*timestep/unit.picoseconds, 1000*elapsed_time/float(nsteps)))

# Time GHMCIntegrator without switching
integrator = GHMCIntegrator(temperature, collision_rate, timestep)
context = create_context(integrator)
initial_time = time.time()
for step in range(nsteps):
    integrator.step(1)
elapsed_time = time.time() - initial_time
del context, integrator
print(format % ('GHMCIntegratorIntegrator', elapsed_time, nsteps, nsteps*timestep/unit.picoseconds, 1000*elapsed_time/float(nsteps)))

# Time GHMCIntegrator with switching
integrator = GHMCIntegrator(temperature, collision_rate, timestep)
context = create_context(integrator)
initial_time = time.time()
for step in range(nsteps):
    set_lambda(context, lambda_value=float(nsteps-step)/float(nsteps))
    integrator.step(1)
elapsed_time = time.time() - initial_time
del context, integrator
print(format % ('GHMCIntegrator with updateParametersInContext', elapsed_time, nsteps, nsteps*timestep/unit.picoseconds, 1000*elapsed_time/float(nsteps)))
