import timeit
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox

size = 10.0
wbox = WaterBox(box_edge=size*unit.angstrom, cutoff=size/2*unit.angstrom, nonbondedMethod=app.PME)
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
context = openmm.Context(wbox.system, integrator,platform,properties)
context.setPositions(wbox.positions)
context.setVelocitiesToTemperature(300*unit.kelvin)
integrator.step(100)

def switchoff(force,context,frac=0.9):
    force.setParticleParameters(0,charge=-0.834*frac,sigma=0.3150752406575124*frac,epsilon=0.635968*frac)
    force.setParticleParameters(1,charge=0.417*frac,sigma=0,epsilon=1*frac)
    force.setParticleParameters(2,charge=0.417*frac,sigma=0,epsilon=1*frac)
    force.updateParametersInContext(context)

def switchon(force,context):
    force.setParticleParameters(0,charge=-0.834,sigma=0.3150752406575124,epsilon=0.635968)
    force.setParticleParameters(1,charge=0.417,sigma=0,epsilon=1)
    force.setParticleParameters(2,charge=0.417,sigma=0,epsilon=1)
    force.updateParametersInContext(context)

force = wbox.system.getForce(2)       # Non-bonded force.

def MDMC(force,context,integrator,steps,i):
    integrator.step(steps)
    if i % 2 == 0:
        switchoff(force,context)
    else:
        switchon(force,context)

def MC(force,context,i):
    if i % 2 == 0:
        switchoff(force,context)
    else:
        switchon(force,context)

# Global parameters for MD and MC iterations
iterations = 200
nsteps = 5000

f = open("timings.txt", 'w')
f.write("iterations={0}, nsteps={1}, size={2}\n".format(iterations,nsteps,size))
f.write("Time in seconds\n")
f.close()
for i in range(iterations):
    f = open("timings.txt","a")
    t = timeit.timeit('MDMC(force,context,integrator,nsteps,i)',setup="from __main__ import MDMC, force, context, integrator, i,nsteps", number=1)
    f.write(str(t)+"\n")
    f.close()
